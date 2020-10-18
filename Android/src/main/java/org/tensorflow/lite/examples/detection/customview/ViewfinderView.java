package org.tensorflow.lite.examples.detection.customview;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.Typeface;
import android.hardware.camera2.CameraManager;
import android.util.AttributeSet;
import android.view.View;

import org.tensorflow.lite.examples.detection.R;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;

/**
 * This view is overlaid on top of the camera preview. It adds the viewfinder rectangle and partial
 * transparency outside it, as well as the laser scanner animation and result points.
 *
 * @author dswitkin@google.com (Daniel Switkin)
 */
public final class ViewfinderView extends View {

    //四个绿色边角对应的长度
    private int ScreenRate;
    //四个绿色边角对应的宽度
    private static final int CORNER_WIDTH = 10;
    //手机的屏幕密度
    private static float density;
    //画笔对象的引用
    private Paint paint;
    private final int maskColor;

    public ViewfinderView(Context context, AttributeSet attrs) {
        super(context, attrs);
        density = context.getResources().getDisplayMetrics().density;
        //像素转化成dp
        ScreenRate = (int) (20 * density);
        paint = new Paint();
        Resources resources = getResources();
        maskColor = resources.getColor(R.color.viewfinder_mask);
    }

    @Override
    protected void onDraw(Canvas canvas) {
        //中间的扫描框,想要修改扫描框的大小可以去CameraManager里面修改
        Frame frame = new Frame(160, 960, 0, 800);
        //获取屏幕的宽和高
        int width = canvas.getWidth();
        int height = canvas.getHeight();
        paint.setColor(maskColor);

        //画出扫描框外面的阴影部分，共四个部分，扫描框的上面到屏幕上面，扫描框的下面到屏幕下面
        //扫描框的左边面到屏幕左边，扫描框的右边到屏幕右边
        // 上
        canvas.drawRect(0, 0, width, frame.top, paint);
        // 左
        canvas.drawRect(0, frame.top, frame.left, frame.bottom + 1, paint);
        // 右
        canvas.drawRect(frame.right + 1, frame.top, width, frame.bottom + 1,
                paint);
        // 下
        canvas.drawRect(0, frame.bottom + 1, width, height, paint);

        //画扫描框边上的角，总共8个部分
        paint.setColor(Color.GREEN);
        canvas.drawRect(frame.left, frame.top, frame.left + ScreenRate,
                frame.top + CORNER_WIDTH, paint);
        canvas.drawRect(frame.left, frame.top, frame.left + CORNER_WIDTH, frame.top
                + ScreenRate, paint);
        canvas.drawRect(frame.right - ScreenRate, frame.top, frame.right,
                frame.top + CORNER_WIDTH, paint);
        canvas.drawRect(frame.right - CORNER_WIDTH, frame.top, frame.right, frame.top
                + ScreenRate, paint);
        canvas.drawRect(frame.left, frame.bottom - CORNER_WIDTH, frame.left
                + ScreenRate, frame.bottom, paint);
        canvas.drawRect(frame.left, frame.bottom - ScreenRate,
                frame.left + CORNER_WIDTH, frame.bottom, paint);
        canvas.drawRect(frame.right - ScreenRate, frame.bottom - CORNER_WIDTH,
                frame.right, frame.bottom, paint);
        canvas.drawRect(frame.right - CORNER_WIDTH, frame.bottom - ScreenRate,
                frame.right, frame.bottom, paint);


    }

    public void drawViewfinder() {
        invalidate();
    }
}


class Frame {
    Frame(int left, int right, int top, int bottom){
        this.left = left;
        this.right = right;
        this.top = top;
        this.bottom = bottom;
    }

    public int left;
    public int right;
    public int top;
    public int bottom;

}