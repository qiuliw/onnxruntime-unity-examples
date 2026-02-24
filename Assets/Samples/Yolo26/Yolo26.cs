using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using Microsoft.ML.OnnxRuntime.Unity;
using Microsoft.ML.OnnxRuntime.UnityEx;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Profiling;
using UnityEngine;
using UnityEngine.Assertions;

namespace Microsoft.ML.OnnxRuntime.Examples
{
    /// <summary>
    /// YOLO26 object detector.
    /// 
    /// This is adapted from the Yolox sample, assuming the same head layout
    /// but with 26 classes instead of 80.
    /// </summary>
    public class Yolo26 : ImageInference<float>
    {
        [Serializable]
        public class Options : ImageInferenceOptions
        {
            [Header("Yolo26 options")]
            public TextAsset labelFile;
            [Range(0f, 1f)]
            public float probThreshold = 0.3f;
            [Range(0f, 1f)]
            public float nmsThreshold = 0.45f;
        }

        public readonly struct Detection : IDetection<Detection>
        {
            public readonly int label;
            public readonly Rect rect;
            public readonly float probability;

            public readonly int Label => label;
            public readonly Rect Rect => rect;

            public Detection(Rect rect, int label, float probability)
            {
                this.rect = rect;
                this.label = label;
                this.probability = probability;
            }

            public int CompareTo(Detection other)
            {
                // Descending order
                return other.probability.CompareTo(probability);
            }
        }

        private readonly struct Anchor
        {
            public readonly int grid0;
            public readonly int grid1;
            public readonly int stride;

            public Anchor(int grid0, int grid1, int stride)
            {
                this.grid0 = grid0;
                this.grid1 = grid1;
                this.stride = stride;
            }

            public static NativeArray<Anchor> GenerateAnchors(int width, int height, Allocator allocator)
            {
                ReadOnlySpan<int> strides = stackalloc int[] { 8, 16, 32 };
                List<Anchor> anchors = new();

                foreach (int stride in strides)
                {
                    int numGridY = height / stride;
                    int numGridX = width / stride;
                    for (int g1 = 0; g1 < numGridY; g1++)
                    {
                        for (int g0 = 0; g0 < numGridX; g0++)
                        {
                            anchors.Add(new Anchor(g0, g1, stride));
                        }
                    }
                }

                return new NativeArray<Anchor>(anchors.ToArray(), allocator);
            }
        }

        public readonly ReadOnlyCollection<string> labelNames;

        // Adjust this constant to match your Yolo26 model's number of classes.
        private const int NUM_CLASSES = 26;

        private readonly NativeArray<Anchor> anchors;
        private readonly Options options;

        private NativeArray<float> output0Native;
        private NativeList<Detection> proposalsList;
        private NativeList<Detection> detectionsList;

        public ReadOnlySpan<Detection> Detections => detectionsList.AsReadOnly();

        static readonly ProfilerMarker generateProposalsMarker = new($"{typeof(Yolo26).Name}.GenerateProposals");

        public Yolo26(byte[] model, Options options)
            : base(model, options)
        {
            this.options = options;

            const int maxDetections = 100;
            const Allocator allocator = Allocator.Persistent;

            proposalsList = new NativeList<Detection>(maxDetections, allocator);
            detectionsList = new NativeList<Detection>(maxDetections, allocator);

            var labelsRaw = options.labelFile.text.Split('\n', StringSplitOptions.RemoveEmptyEntries);
            for (int i = 0; i < labelsRaw.Length; i++)
            {
                labelsRaw[i] = labelsRaw[i].Trim();
            }
            labelNames = Array.AsReadOnly(labelsRaw);
            Assert.AreEqual(NUM_CLASSES, labelNames.Count);

            anchors = Anchor.GenerateAnchors(Width, Height, allocator);
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                output0Native.Dispose();
                proposalsList.Dispose();
                detectionsList.Dispose();
                anchors.Dispose();
            }
            base.Dispose(disposing);
        }

        protected override void PostProcess(IReadOnlyList<OrtValue> outputs)
        {
            var output0 = outputs[0].GetTensorDataAsSpan<float>();

            generateProposalsMarker.Begin();
            var handle = GenerateProposals(output0, proposalsList, options.probThreshold);
            handle.Complete();
            generateProposalsMarker.End();

            proposalsList.Sort();
            DetectionUtil.NMS(proposalsList, detectionsList, options.nmsThreshold);
        }

        /// <summary>
        /// Convert CV rect to Viewport space.
        /// </summary>
        public Rect ConvertToViewport(in Rect rect)
        {
            Rect unityRect = rect.FlipY();
            var mtx = InputToViewportMatrix;
            Vector2 min = mtx.MultiplyPoint3x4(unityRect.min);
            Vector2 max = mtx.MultiplyPoint3x4(unityRect.max);
            return new Rect(min, max - min);
        }

        private JobHandle GenerateProposals(
            in ReadOnlySpan<float> featBlob,
            NativeList<Detection> result,
            float probThreshold)
        {
            result.Clear();

            if (!output0Native.IsCreated)
            {
                output0Native = new NativeArray<float>(featBlob.Length, Allocator.Persistent);
            }
            featBlob.CopyTo(output0Native.AsSpan());

            var job = new GenerateProposalsJob
            {
                anchors = anchors,
                featBlob = output0Native,
                widthScale = 1f / Width,
                heightScale = 1f / Height,
                probThreshold = probThreshold,
                proposals = result.AsParallelWriter()
            };
            return job.Schedule(anchors.Length, 64);
        }

        [BurstCompile]
        private struct GenerateProposalsJob : IJobParallelFor
        {
            [ReadOnly]
            public NativeArray<Anchor> anchors;

            // shape: 1, N, 5 + NUM_CLASSES
            [ReadOnly]
            public NativeArray<float> featBlob;

            public float widthScale;
            public float heightScale;
            public float probThreshold;

            [WriteOnly]
            public NativeList<Detection>.ParallelWriter proposals;

            public void Execute(int anchorId)
            {
                var anchor = anchors[anchorId];
                int grid0 = anchor.grid0;
                int grid1 = anchor.grid1;
                int stride = anchor.stride;

                int basicPos = anchorId * (NUM_CLASSES + 5);

                float xCenter = (featBlob[basicPos + 0] + grid0) * stride;
                float yCenter = (featBlob[basicPos + 1] + grid1) * stride;
                float w = math.exp(featBlob[basicPos + 2]) * stride;
                float h = math.exp(featBlob[basicPos + 3]) * stride;

                xCenter *= widthScale;
                yCenter *= heightScale;
                w *= widthScale;
                h *= heightScale;

                if (xCenter < 0 || xCenter > 1 || yCenter < 0 || yCenter > 1)
                {
                    return;
                }

                float x0 = xCenter - w * 0.5f;
                float y0 = yCenter - h * 0.5f;

                float boxObjectness = featBlob[basicPos + 4];
                for (int classIdx = 0; classIdx < NUM_CLASSES; classIdx++)
                {
                    float boxClsScore = featBlob[basicPos + 5 + classIdx];
                    float boxProb = boxObjectness * boxClsScore;
                    if (boxProb > probThreshold)
                    {
                        proposals.AddNoResize(new Detection(
                            new Rect(x0, y0, w, h),
                            classIdx,
                            boxProb
                        ));
                    }
                }
            }
        }
    }
}

