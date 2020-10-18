/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.detection;

import android.app.Activity;
import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.hardware.Camera;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Build;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.util.FloatMath;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Toast;

import androidx.annotation.RequiresApi;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;
import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.customview.OverlayView.DrawCallback;
import org.tensorflow.lite.examples.detection.customview.SRImageView;
import org.tensorflow.lite.examples.detection.customview.ViewfinderView;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Classifier;
import org.tensorflow.lite.examples.detection.tflite.SRModel;
import org.tensorflow.lite.examples.detection.tflite.TFLiteObjectDetectionAPIModel;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 * 模型主类
 */
public class DetectorActivity extends CameraActivity implements
        OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();

  // 超分辨配置
  private static final String SR_delegate = "GPU";
  private static final String SR_tflite_model_file = "mobile_unet_200.tflite";
  private static final boolean SR_quantized = false;
  private SRModel sr_model;
  private boolean is_sr =false;

  //实现双击进行超分辨
  private int click_count = 0;
  private long down_click = 0;
  private long up_click = 0;
  private final int interval_time = 500;


  // Configuration values for the prepackaged SSD model.
  private static final int TF_OD_API_INPUT_SIZE = 800; //300
  private static final boolean TF_OD_API_IS_QUANTIZED = false;
  private static final String TF_OD_API_MODEL_FILE = "surf_0620_origin-9_f32.tflite";
  private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/cervical.txt";
  private static final String Detection_delegate = "GPU";


  private static final DetectorMode MODE = DetectorMode.TF_OD_API;
  // Minimum detection confidence to track a detection.
  public static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.6f;
  private static final boolean MAINTAIN_ASPECT = false;
  private static final Size DESIRED_PREVIEW_SIZE = new Size(800, 800);
  private static final boolean SAVE_PREVIEW_BITMAP = false;
  private static final float TEXT_SIZE_DIP = 5;
  OverlayView trackingOverlay;
  ViewfinderView finderView;
  private Integer sensorOrientation;

  private Classifier detector;

  private long lastProcessingTimeMs;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap cropCopyBitmap = null;
  private Bitmap srBitmap = null;

  private boolean computingDetection = false;

  private long timestamp = 0;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  private MultiBoxTracker tracker;

  private BorderedText borderedText;

  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    final float textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    tracker = new MultiBoxTracker(this);

    int cropSize = TF_OD_API_INPUT_SIZE;

    // 初始化超分辨模型
    initSrModel();
    try {
      detector =
          TFLiteObjectDetectionAPIModel.create(
              getAssets(),
              TF_OD_API_MODEL_FILE,
              TF_OD_API_LABELS_FILE,
              TF_OD_API_INPUT_SIZE,
              TF_OD_API_IS_QUANTIZED, Detection_delegate);
      cropSize = TF_OD_API_INPUT_SIZE;
      Log.d("YoloV4Detector", "Initializing SUCCESS !!");
    } catch (final IOException e) {
      e.printStackTrace();
      LOGGER.e(e, "Exception initializing classifier!");
      Toast toast =
          Toast.makeText(
              getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
      toast.show();
      finish();
    }


    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    sensorOrientation = rotation - getScreenOrientation();
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
//    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    rgbFrameBitmap = Bitmap.createBitmap(800, 800, Config.ARGB_8888);
    croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);
    srBitmap = Bitmap.createBitmap(200, 200, Config.ARGB_8888);
    // 从 frame到Crop的仿射变换
    frameToCropTransform =
        ImageUtils.getTransformationMatrix(
            800, 800,
            cropSize, cropSize,
            sensorOrientation, MAINTAIN_ASPECT);

    // 从crop到Frame的仿射变化
    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);

    finderView = findViewById(R.id.finder_view);
    finderView.drawViewfinder();
    trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
    //检测回调
    trackingOverlay.addCallback(
        new DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            tracker.draw(canvas);
            if (isDebug()) {
              tracker.drawDebug(canvas);
            }
          }
        });

    tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
  }

  final int[] nnnnn = {0};
  @Override
  protected void processImage() {
    ++timestamp;
    final long currTimestamp = timestamp;
    trackingOverlay.postInvalidate();

    // No mutex needed as this method is not reentrant.
    if (computingDetection) {
      readyForNextImage();
      return;
    }
    // SR中保持画面
//    if (is_sr){
//      readyForNextImage();
//      return;
//    }

    computingDetection = true;
    LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

//    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);
    rgbFrameBitmap.setPixels(getRgbBytes(), 0, 800, 0, 0, 800, 800);

    readyForNextImage();

    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }

    runInBackground(
        new Runnable() {
          @Override
          public void run() {
            LOGGER.i("Running detection on image " + currTimestamp);
            final long startTime = SystemClock.uptimeMillis();

            final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

            //拷贝图像，并叠加矩形检测框后显示
            cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
            final Canvas canvas = new Canvas(cropCopyBitmap);
            final Paint paint = new Paint();
            paint.setColor(Color.RED);
            paint.setStyle(Style.STROKE);
            paint.setStrokeWidth(2.0f);

            float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
            switch (MODE) {
              case TF_OD_API:
                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                break;
            }

            final List<Classifier.Recognition> mappedRecognitions =
                new LinkedList<Classifier.Recognition>();

            for (final Classifier.Recognition result : results) {
              final RectF location = result.getLocation();
              if (location != null && result.getConfidence() >= minimumConfidence) {
                canvas.drawRect(location, paint);

                cropToFrameTransform.mapRect(location);

                result.setLocation(location);
                mappedRecognitions.add(result);
              }
            }

            //画检测框
            tracker.trackResults(mappedRecognitions, currTimestamp);
//            tracker.trackSRImage(cropCopyBitmap);
            trackingOverlay.postInvalidate();

            computingDetection = false;
            runOnUiThread(
                new Runnable() {
                  @Override
                  public void run() {
                    showFrameInfo(previewWidth + "x" + previewHeight);
                    showCropInfo(rgbFrameBitmap.getWidth() + "x" + rgbFrameBitmap.getHeight());
                    showInference(lastProcessingTimeMs + "ms");
//                    Bitmap sr_input = Bitmap.createBitmap(croppedBitmap, 0,0, 100, 100,null, false);
//                    setSRImageView(sr_input);
                  }
                });
          }
        });

  }

  @Override
  protected int getLayoutId() {
    return R.layout.tfe_od_camera_connection_fragment_tracking;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  // Which detection model to use: by default uses Tensorflow Object Detection API frozen
  // checkpoints.
  private enum DetectorMode {
    TF_OD_API;
  }

  @Override
  protected void setUseNNAPI(final boolean isChecked) {
    runInBackground(() -> detector.setUseNNAPI(isChecked));
  }

  @Override
  protected void setNumThreads(final int numThreads) {
    runInBackground(() -> detector.setNumThreads(numThreads));
  }

  // 超分辨setting
  private void initSrModel()
  {
    try{
      AssetManager assetManager = getAssets();
      sr_model = SRModel.create(assetManager, SR_tflite_model_file, SR_quantized, SR_delegate);
    }
    catch(final IOException e)
    {
      e.printStackTrace();
      Toast toast =
              Toast.makeText(
                      getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
      toast.show();
      finish();
    }
  }

  //////// 触摸屏幕触发超分辨 deleted
  // 触摸屏幕获取待预测图像
  final double det2sr_scale = 0.4284 / 0.6426;
  @RequiresApi(api = Build.VERSION_CODES.M)
  @Override
  public boolean onTouchEvent(MotionEvent event){
    int statusBarHeight1 = 0;
    //获取status_bar_height资源的ID
    int resourceId = getResources().getIdentifier("status_bar_height", "dimen", "android");
    if (resourceId > 0) {
      //根据资源ID获取响应的尺寸值
      statusBarHeight1 = getResources().getDimensionPixelSize(resourceId);
    }
    int finalStatusBarHeight = statusBarHeight1;// 超分辨处理
    RectF valid_std_rect = new RectF(0, 0, 800, 800);
    RectF sr_std_rect = new RectF(0, 0, 200, 200);
    final float validRange_h = valid_std_rect.height();
    final float validRange_w = valid_std_rect.width();
    final double sr_h = sr_std_rect.height() * det2sr_scale;
    final double sr_w = sr_std_rect.width() * det2sr_scale;
    final double sr_half_h = sr_h / 2;
    final double sr_half_w = sr_w / 2;
    float x = previewHeight-(float) event.getX();
    float y = (float) event.getY()-2*finalStatusBarHeight;
    RectF sr_point = new RectF((int) y, (int) x, (int) y, (int) x);
    frameToCropTransform.mapRect(sr_point);
    int point_x_start = (int) sr_point.centerX() - (int) sr_half_w;
    int point_y_start = (int) sr_point.centerY() - (int) sr_half_h;
    int point_x_end = point_x_start + (int) sr_w;
    int point_y_end = point_y_start + (int) sr_h;
    int warpped_x_start = (int) sr_point.centerX() - (int) sr_w;
    int warpped_y_start = (int) sr_point.centerY() - (int) sr_half_h;
    int warpped_x_end = warpped_x_start + (int) sr_w;
    int warpped_y_end = warpped_x_start + (int) sr_h;
    warpped_x_start = Math.max(warpped_x_start, 0);
    warpped_y_start = Math.max(warpped_y_start, 0);
    warpped_x_start = warpped_x_end <= validRange_w ? warpped_x_start : (int) validRange_w - (int) sr_w;
    warpped_y_start = warpped_y_end <= validRange_h ? warpped_y_start : (int) validRange_h - (int) sr_h;
    if (point_x_start >= 0 && point_x_end <= validRange_w && point_y_start >= 0 && point_y_end <= validRange_h) {
        srInput = Bitmap.createBitmap(croppedBitmap, warpped_x_start, warpped_y_start, (int) sr_w, (int) sr_h, null, false);
        srInput = SRImageView.resizeImage(srInput, 200, 200);
    }
    setSRImageView(srInput);
    setOrigImageView(srInput);
    setSRLog("touched point: X="+event.getX()+", Y="+event.getY()+"\nwarpped point:  X="+warpped_x_start+", Y="+warpped_y_start);
    return true;
  }

}
