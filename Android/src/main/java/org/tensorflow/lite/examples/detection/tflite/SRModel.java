package org.tensorflow.lite.examples.detection.tflite;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.Map;

public class SRModel {
    // Float model
//    private static final float IMAGE_MEAN = 128.0f;
//    private static final float IMAGE_STD = 128.0f;
    private static final float IMAGE_MEAN = 0.0f;
    private static final float IMAGE_STD = 255.0f;
    private static final int NUM_THREADS = 4;
    //model input size
    private int m_input_height = 200;
    private int m_input_width = 200;
    //output SR images, [N,H,W,C]
    private int m_output_height = m_input_height*2;
    private int m_output_width = m_input_width*2;
    private int m_output_channel = 3;
    private float[][][][] m_output_img;

    private String delegate;
    private boolean isModelQuantized;
    private int[] intValues;

    private ByteBuffer imgData;

    private Interpreter tfLite;


    public static SRModel create(
            final AssetManager assetManager,
            final String modelFilename,
            final boolean isQuantized,
            final String device) throws IOException
    {
        final SRModel d = new SRModel();
        try {
            if (device == "GPU"){
                GpuDelegate delegate = new GpuDelegate();
                Interpreter.Options options = (new Interpreter.Options()).addDelegate(delegate);
                // setup interpreter
                d.tfLite = new Interpreter(loadModelFile(assetManager, modelFilename), options);
            }
            else if (device =="NNAPI"){
                Interpreter.Options options = (new Interpreter.Options());
                NnApiDelegate nnApiDelegate = null;
                d.tfLite = new Interpreter(loadModelFile(assetManager, modelFilename), options);
            }
            else {
                d.tfLite = new Interpreter(loadModelFile(assetManager, modelFilename));
            }
//            GpuDelegate delegate = new GpuDelegate();
//            Interpreter.Options options = (new Interpreter.Options()).addDelegate(delegate);
//            d.tfLite = new Interpreter(loadModelFile(assetManager, modelFilename), options);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        d.isModelQuantized = isQuantized;
        d.delegate = device;
        // Pre-allocate buffers.
        int numBytesPerChannel;
        if (isQuantized) {
            numBytesPerChannel = 1; // Quantized
        } else {
            numBytesPerChannel = 4; // Floating point, 送入模型的是否是float
        }
        d.imgData = ByteBuffer.allocateDirect(1 * d.m_input_height * d.m_input_width * 3 * numBytesPerChannel);
        d.imgData.order(ByteOrder.nativeOrder());
        d.intValues = new int[d.m_input_height * d.m_input_width];
        d.tfLite.setNumThreads(NUM_THREADS);
        d.m_output_img = new float[1][d.m_output_height][d.m_output_width][d.m_output_channel];
        return d;
    }

    public Bitmap superResolutionImg(Bitmap bitmap)
    {
        bitmap = Bitmap.createScaledBitmap(bitmap, m_input_width, m_input_height, false);
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        // not understand yet,
        imgData.rewind();
        for (int i = 0; i < m_input_width; ++i) {
            for (int j = 0; j < m_input_height; ++j) {
                int pixelValue = intValues[i * m_input_height + j];
                if (isModelQuantized) {
                    // Quantized model
                    imgData.put((byte) ((pixelValue >> 16) & 0xFF));
                    imgData.put((byte) ((pixelValue >> 8) & 0xFF));
                    imgData.put((byte) (pixelValue & 0xFF));
                } else { // Float model
                    int r = (pixelValue>>16)&0xFF;
                    int g = (pixelValue>>8)&0xFF;
                    int b = pixelValue&0xFF;
                    float r_float = (r - IMAGE_MEAN)/IMAGE_STD;
                    float g_float = (g - IMAGE_MEAN)/IMAGE_STD;
                    float b_float = (b - IMAGE_MEAN)/IMAGE_STD;
                    imgData.putFloat(r_float);
                    imgData.putFloat(g_float);
                    imgData.putFloat(b_float);
//                    imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
//                    imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
//                    imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                }
            }
        }
        Object [] inputArray = {imgData};
        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, m_output_img);
        tfLite.runForMultipleInputsOutputs(inputArray, outputMap);
//        for(int i = 0;i < m_output_width;i++)
//        {
//            for(int j = 0;j < m_output_height;j++)
//            {
//                float r = m_output_img[0][i][j][0];
//                float g = m_output_img[0][i][j][1];
//                float b = m_output_img[0][i][j][2];
//            }
//        }
        Bitmap resultBitmap = this.convertArray2Bitmap(m_output_img, m_output_width, m_output_height);
        return resultBitmap;
    }

    public Bitmap convertArray2Bitmap(float data[][][][], int width, int height){

        int castdata[] = new int[width * height];
        for (int i=0;i<width;i++){
            for (int j=0;j<height;j++){
                float float_r = data[0][i][j][0];
                float float_g = data[0][i][j][1];
                float float_b = data[0][i][j][2];

                int r = Math.round(float_r * 255f);
                int g = Math.round(float_g * 255f);
                int b = Math.round(float_b * 255f);
                castdata[i*height+j] = 0xFF000000 | ((r&0xFF)<<16) | ((g&0xFF)<<8) | b;
            }
        }
        Bitmap bitmap = Bitmap.createBitmap(castdata, 0, width,width,height, Bitmap.Config.ARGB_8888);

        return  bitmap;
    }

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

}
