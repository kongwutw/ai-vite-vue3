<script setup lang="ts">
import { onMounted } from 'vue';
// import {InferenceSession, Tensor} from 'onnxruntime-web';
import * as ort from "onnxruntime-web";
import loadImage from "blueimp-load-image";
import ndarray from "ndarray";
import ops from "ndarray-ops";

const preprocess = (ctx: CanvasRenderingContext2D) => {
    const imageData = ctx.getImageData(
      0,
      0,
      ctx.canvas.width,
      ctx.canvas.height
    );
    const { data, width, height } = imageData;

    // data processing
    const dataTensor = ndarray(new Float32Array(data), [width, height, 4]);
    const dataProcessedTensor = ndarray(new Float32Array(width * height * 3), [
      1,
      3,
      width,
      height,
    ]);

    ops.assign(
      dataProcessedTensor.pick(0, 0, null, null),
      dataTensor.pick(null, null, 0)
    );
    ops.assign(
      dataProcessedTensor.pick(0, 1, null, null),
      dataTensor.pick(null, null, 1)
    );
    ops.assign(
      dataProcessedTensor.pick(0, 2, null, null),
      dataTensor.pick(null, null, 2)
    );

    ops.divseq(dataProcessedTensor, 255);
    ops.subseq(dataProcessedTensor.pick(0, 0, null, null), 0.485);
    ops.subseq(dataProcessedTensor.pick(0, 1, null, null), 0.456);
    ops.subseq(dataProcessedTensor.pick(0, 2, null, null), 0.406);

    ops.divseq(dataProcessedTensor.pick(0, 0, null, null), 0.229);
    ops.divseq(dataProcessedTensor.pick(0, 1, null, null), 0.224);
    ops.divseq(dataProcessedTensor.pick(0, 2, null, null), 0.225);

    const tensor = new ort.Tensor("float32", new Float32Array(width * height * 3), [
      1,
      3,
      width,
      height,
    ]);
    (tensor.data as Float32Array).set(dataProcessedTensor.data);
    return tensor;
  }

onMounted(async() => {
  const response = await fetch('/mobilenetv2-7.onnx');
  const modelFile = await response.arrayBuffer();
  const session = await ort.InferenceSession.create(modelFile, { executionProviders: ['webgl'] });
  setTimeout(async() => {
    const dims = [ 1, 3, 224, 224];
    const size = dims.reduce((a, b) => a * b);
    const warmupTensor = new ort.Tensor('float32', new Float32Array(size), dims);
    for (let i = 0; i < size; i++) {
      warmupTensor.data[i] = Math.random() * 2.0 - 1.0;
    }
    const feeds: Record<string, ort.Tensor> = {};
    feeds[session.inputNames[0]] = warmupTensor;
    await session.run(feeds);
    
    loadImage('/bird.jpg', (img) => {
      const element = document.getElementById("input-canvas") as HTMLCanvasElement;
      if (element) {
        const ctx = element.getContext("2d");
        if (ctx) {
          ctx.drawImage(img, 0, 0);
          setTimeout(async() => {
            const element = document.getElementById("input-canvas") as HTMLCanvasElement;
            const ctx = element.getContext("2d") as CanvasRenderingContext2D;
            const preprocessedData = preprocess(ctx);
            const feeds: Record<string, ort.Tensor> = {};
            feeds[session.inputNames[0]] = preprocessedData;
            const outputData = await session.run(feeds);
            const output = outputData[session.outputNames[0]];
            console.log('111', output, session.outputNames[0], outputData);
          }, 10);
        }
      }
    });
  }, 0);

});
</script>

<template>
  <div>
    <h1>demo</h1>
    <canvas id="input-canvas" width="224" height="224" />
  </div>
</template>
