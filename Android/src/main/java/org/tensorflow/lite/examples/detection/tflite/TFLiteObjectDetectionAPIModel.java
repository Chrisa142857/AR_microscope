/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.lite.examples.detection.tflite;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Trace;
import android.util.Log;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Vector;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.examples.detection.DetectorActivity;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;

import static org.tensorflow.lite.examples.detection.env.Utils.expit;

/**
 * Wrapper for frozen detection models trained using the Tensorflow Object Detection API:
 * - https://github.com/tensorflow/models/tree/master/research/object_detection
 * where you can find the training code.
 *
 * To use pretrained models in the API or convert to TF Lite models, please see docs for details:
 * - https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
 * - https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md#running-our-model-on-android
 */
public class TFLiteObjectDetectionAPIModel implements Classifier {
  private static final Logger LOGGER = new Logger();
  //YOLO config
  //config yolov4
  private static final int[] OUTPUT_WIDTH = new int[]{25, 50};
  private static final int[] actual_width = new int[]{25, 50};

  private static final int NUM_BOXES_PER_BLOCK = 3;
  protected static final int BATCH_SIZE = 1;
  protected static final int PIXEL_SIZE = 3;


  // Number of threads in the java app
  private String delegate;
  private static final int NUM_THREADS = 4;
  private boolean isModelQuantized;
  // Config values.
  private int inputSize;
  // Pre-allocated buffers.
  private Vector<String> labels = new Vector<String>();

  private ByteBuffer imgData;

  private Interpreter tfLite;

  private TFLiteObjectDetectionAPIModel() {}

  /** Memory-map the model file in Assets. */
  private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
      throws IOException {
    AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }

  /**
   * Initializes a native TensorFlow session for classifying images.
   *
   * @param assetManager The asset manager to be used to load assets.
   * @param modelFilename The filepath of the model GraphDef protocol buffer.
   * @param labelFilename The filepath of label file for classes.
   * @param inputSize The size of image input
   * @param isQuantized Boolean representing model is quantized or not
   */
  public static Classifier create(
      final AssetManager assetManager,
      final String modelFilename,
      final String labelFilename,
      final int inputSize,
      final boolean isQuantized,
      String device)
      throws IOException {
    final TFLiteObjectDetectionAPIModel d = new TFLiteObjectDetectionAPIModel();

    String actualFilename = labelFilename.split("file:///android_asset/")[1];
    InputStream labelsInput = assetManager.open(actualFilename);
    BufferedReader br = new BufferedReader(new InputStreamReader(labelsInput));
    String line;
    while ((line = br.readLine()) != null) {
      LOGGER.w(line);
      d.labels.add(line);
    }
    br.close();

    d.inputSize = inputSize;

    try {
      if (device.equals("GPU")){
        GpuDelegate delegate = new GpuDelegate();
        Interpreter.Options options = (new Interpreter.Options()).addDelegate(delegate);
        // setup interpreter
        MappedByteBuffer model  = loadModelFile(assetManager, modelFilename);
        d.tfLite = new Interpreter(model, options);
      }
      else if (device.equals("NNAPI")){
        Interpreter.Options options = (new Interpreter.Options());
        NnApiDelegate nnApiDelegate = null;
        d.tfLite = new Interpreter(loadModelFile(assetManager, modelFilename), options);
      }
      else
        d.tfLite = new Interpreter(loadModelFile(assetManager, modelFilename));
    } catch (Exception e) {
      throw new RuntimeException(e);
    }

    d.isModelQuantized = isQuantized;
    // Pre-allocate buffers.
    int numBytesPerChannel;
    if (isQuantized) {
      numBytesPerChannel = 1; // Quantized
    } else {
      numBytesPerChannel = 4; // Floating point
    }
    d.imgData = ByteBuffer.allocateDirect(1 * d.inputSize * d.inputSize * 3 * numBytesPerChannel);
    d.imgData.order(ByteOrder.nativeOrder());
    d.tfLite.setNumThreads(NUM_THREADS);
    d.delegate = device;
    return d;
  }
  /*
  @Override
  public List<Recognition> recognizeImage(final Bitmap bitmap) {
    // Log this method so that it can be analyzed with systrace.
    Trace.beginSection("recognizeImage");

    Trace.beginSection("preprocessBitmap");
    // Preprocess the image data from 0-255 int to normalized float based
    // on the provided parameters.
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

    imgData.rewind();
    for (int i = 0; i < inputSize; ++i) {
      for (int j = 0; j < inputSize; ++j) {
        int pixelValue = intValues[i * inputSize + j];
        if (isModelQuantized) {
          // Quantized model
          imgData.put((byte) ((pixelValue >> 16) & 0xFF));
          imgData.put((byte) ((pixelValue >> 8) & 0xFF));
          imgData.put((byte) (pixelValue & 0xFF));
        } else { // Float model
          imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
          imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
          imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
        }
      }
    }
    Trace.endSection(); // preprocessBitmap

    // Copy the input data into TensorFlow.
    Trace.beginSection("feed");
    outputLocations = new float[1][NUM_DETECTIONS][4];
    outputClasses = new float[1][NUM_DETECTIONS];
    outputScores = new float[1][NUM_DETECTIONS];
    numDetections = new float[1];

    Object[] inputArray = {imgData};
    Map<Integer, Object> outputMap = new HashMap<>();
    outputMap.put(0, outputLocations);
    outputMap.put(1, outputClasses);
    outputMap.put(2, outputScores);
    outputMap.put(3, numDetections);
    Trace.endSection();

    // Run the inference call.
    Trace.beginSection("run");
    tfLite.runForMultipleInputsOutputs(inputArray, outputMap);
    Trace.endSection();

    // Show the best detections.
    // after scaling them back to the input size.

    // You need to use the number of detections from the output and not the NUM_DETECTONS variable declared on top
      // because on some models, they don't always output the same total number of detections
      // For example, your model's NUM_DETECTIONS = 20, but sometimes it only outputs 16 predictions
      // If you don't use the output's numDetections, you'll get nonsensical data
    int numDetectionsOutput = Math.min(NUM_DETECTIONS, (int) numDetections[0]); // cast from float to integer, use min for safety

    final ArrayList<Recognition> recognitions = new ArrayList<>(numDetectionsOutput);
    for (int i = 0; i < numDetectionsOutput; ++i) {
      final RectF detection =
          new RectF(
              outputLocations[0][i][1] * inputSize,
              outputLocations[0][i][0] * inputSize,
              outputLocations[0][i][3] * inputSize,
              outputLocations[0][i][2] * inputSize);
      // SSD Mobilenet V1 Model assumes class 0 is background class
      // in label file and class labels start from 1 to number_of_classes+1,
      // while outputClasses correspond to class index from 0 to number_of_classes
      int labelOffset = 1;
      recognitions.add(
          new Recognition(
              "" + i,
              labels.get((int) outputClasses[0][i] + labelOffset),
              outputScores[0][i],
              detection));
    }
    Trace.endSection(); // "recognizeImage"
    return recognitions;
  }
  */

  public ArrayList<Recognition> recognizeImage(Bitmap bitmap) {
    ByteBuffer byteBuffer = convertBitmapToByteBuffer(bitmap);

    Map<Integer, Object> outputMap = new HashMap<>();
    for (int i = 0; i < OUTPUT_WIDTH.length; i++) {
      float[][][][][] out = new float[1][OUTPUT_WIDTH[i]][OUTPUT_WIDTH[i]][3][5 + labels.size()];
      outputMap.put(i, out);
    }

    Log.d("YoloV4Classifier", "mObjThresh: " + getObjThresh());

    Object[] inputArray = {byteBuffer};
    tfLite.runForMultipleInputsOutputs(inputArray, outputMap);

    ArrayList<Recognition> detections = new ArrayList<Recognition>();

    for (int i = 0; i < actual_width.length; i++) {
      int gridWidth = actual_width[i];
      float[][][][][] out = (float[][][][][]) outputMap.get(i);

      Log.d("YoloV4Classifier", "out[" + i + "] detect start");
      for (int y = 0; y < gridWidth; ++y) {
        for (int x = 0; x < gridWidth; ++x) {
          for (int b = 0; b < NUM_BOXES_PER_BLOCK; ++b) {
            final int offset =
                    (gridWidth * (NUM_BOXES_PER_BLOCK * (labels.size() + 5))) * y
                            + (NUM_BOXES_PER_BLOCK * (labels.size() + 5)) * x
                            + (labels.size() + 5) * b;

            final float confidence = expit(out[0][y][x][b][4]);
            int detectedClass = -1;
            float maxClass = 0;

            final float[] classes = new float[labels.size()];
            for (int c = 0; c < labels.size(); ++c) {
              classes[c] = out[0][y][x][b][5 + c];
            }

            for (int c = 0; c < labels.size(); ++c) {
              if (classes[c] > maxClass) {
                detectedClass = c;
                maxClass = classes[c];
              }
            }

            final float confidenceInClass = maxClass * confidence;
            if (confidenceInClass > getObjThresh()) {

              final float xPos = out[0][y][x][b][0];
              final float yPos = out[0][y][x][b][1];

              final float w = out[0][y][x][b][2];
              final float h = out[0][y][x][b][3];

              final RectF rect =
                      new RectF(
                              Math.max(0, xPos - w / 2),
                              Math.max(0, yPos - h / 2),
                              Math.min(bitmap.getWidth() - 1, xPos + w / 2),
                              Math.min(bitmap.getHeight() - 1, yPos + h / 2));
              detections.add(new Recognition("" + offset, labels.get(detectedClass),
                      confidenceInClass, rect, detectedClass));
            }
          }
        }
      }
      Log.d("YoloV4Classifier", "out[" + i + "] detect end");
    }

    final ArrayList<Recognition> recognitions = nms(detections);

    return recognitions;
//      return detections;
  }
  /**
   * Writes Image data into a {@code ByteBuffer}.
   */
  protected ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
    ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * BATCH_SIZE * inputSize * inputSize* PIXEL_SIZE);
    byteBuffer.order(ByteOrder.nativeOrder());
    int[] intValues = new int[inputSize * inputSize];
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
    int pixel = 0;
    for (int i = 0; i < inputSize; ++i) {
      for (int j = 0; j < inputSize; ++j) {
        final int val = intValues[pixel++];
        byteBuffer.putFloat(((val >> 16) & 0xFF) / 255.0f);
        byteBuffer.putFloat(((val >> 8) & 0xFF) / 255.0f);
        byteBuffer.putFloat((val & 0xFF) / 255.0f);
      }
    }
    return byteBuffer;
  }

  @Override
  public void enableStatLogging(final boolean logStats) {}

  @Override
  public String getStatString() {
    return "";
  }

  @Override
  public void close() {}

  public void setNumThreads(int num_threads) {
    if (tfLite != null) tfLite.setNumThreads(num_threads);
  }

  @Override
  public void setUseNNAPI(boolean isChecked) {
    if (tfLite != null) tfLite.setUseNNAPI(isChecked);
  }

  public float getObjThresh(){
    return DetectorActivity.MINIMUM_CONFIDENCE_TF_OD_API;
  }

  //non maximum suppression
  protected ArrayList<Recognition> nms(ArrayList<Recognition> list) {
    ArrayList<Recognition> nmsList = new ArrayList<Recognition>();

    for (int k = 0; k < labels.size(); k++) {
      //1.find max confidence per class
      PriorityQueue<Recognition> pq =
              new PriorityQueue<Recognition>(
                      50,
                      new Comparator<Recognition>() {
                        @Override
                        public int compare(final Recognition lhs, final Recognition rhs) {
                          // Intentionally reversed to put high confidence at the head of the queue.
                          return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                        }
                      });

      for (int i = 0; i < list.size(); ++i) {
        if (list.get(i).getDetectedClass() == k) {
          pq.add(list.get(i));
        }
      }

      //2.do non maximum suppression
      while (pq.size() > 0) {
        //insert detection with max confidence
        Recognition[] a = new Recognition[pq.size()];
        Recognition[] detections = pq.toArray(a);
        Recognition max = detections[0];
        nmsList.add(max);
        pq.clear();

        for (int j = 1; j < detections.length; j++) {
          Recognition detection = detections[j];
          RectF b = detection.getLocation();
          if (box_iou(max.getLocation(), b) < mNmsThresh) {
            pq.add(detection);
          }
        }
      }
    }
    return nmsList;
  }

  protected float mNmsThresh = 0.2f;

  protected float box_iou(RectF a, RectF b) {
    return box_intersection(a, b) / box_union(a, b);
  }

  protected float box_intersection(RectF a, RectF b) {
    float w = overlap((a.left + a.right) / 2, a.right - a.left,
            (b.left + b.right) / 2, b.right - b.left);
    float h = overlap((a.top + a.bottom) / 2, a.bottom - a.top,
            (b.top + b.bottom) / 2, b.bottom - b.top);
    if (w < 0 || h < 0) return 0;
    float area = w * h;
    return area;
  }

  protected float box_union(RectF a, RectF b) {
    float i = box_intersection(a, b);
    float u = (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - i;
    return u;
  }

  protected float overlap(float x1, float w1, float x2, float w2) {
    float l1 = x1 - w1 / 2;
    float l2 = x2 - w2 / 2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1 / 2;
    float r2 = x2 + w2 / 2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
  }
}
