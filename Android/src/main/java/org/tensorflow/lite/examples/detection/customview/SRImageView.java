package org.tensorflow.lite.examples.detection.customview;

import android.content.Context;
import android.database.ContentObserver;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.drawable.BitmapDrawable;
import android.util.AttributeSet;

import androidx.annotation.Nullable;

public class SRImageView extends androidx.appcompat.widget.AppCompatImageView {
    public SRImageView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
    }
    private SRCallback srCallback;
    public void setOnContentAvailable(SRCallback srCallback) {
        this.srCallback = srCallback;
    }
    public Bitmap proposedImage;
    public void setProposedImage(Bitmap bitmap){
        this.proposedImage = bitmap;
    }
    public void setBitmap(){
        if (this.proposedImage != null) {
            this.setImageBitmap(resizeImage(this.proposedImage, (int) (400), (int) (400)));
        } else {
            this.setImageBitmap(null);
        }
    }

    static public Bitmap resizeImage(Bitmap bitmap, int w, int h) {
        Bitmap BitmapOrg = bitmap;
        int width = BitmapOrg.getWidth();
        int height = BitmapOrg.getHeight();
        int newWidth = w;
        int newHeight = h;

        float scaleWidth = ((float) newWidth) / width;
        float scaleHeight = ((float) newHeight) / height;

        Matrix matrix = new Matrix();
        matrix.postScale(scaleWidth, scaleHeight);
        // if you want to rotate the Bitmap
        // matrix.postRotate(45);
        Bitmap resizedBitmap = Bitmap.createBitmap(BitmapOrg, 0, 0, width,
                height, matrix, true);
        return resizedBitmap;
    }

    @Override
    protected void onSizeChanged(int w, int h, int oldw, int oldh) {
        super.onSizeChanged(w, h, oldw, oldh);
//        srCallback.onContentAvailable(this);
//        this.setBitmap();
    }

    /** Interface defining the callback for client classes. */
    public interface SRCallback {
        void onContentAvailable(SRImageView srImageView);
    }
}

