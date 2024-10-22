#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;

void intro()
{
	Mat im = imread("eper.jpg", 1);
	imshow("Ez itt egy alma", im);
	waitKey(0);
}

void lab01()
{
	Mat im1 = imread("3.jpg", 1);
	Mat im2 = imread("5.jpg", 1);
	imshow("Film", im1);
	waitKey(0);
	Mat im3 = im2.clone();
    while (true) {
        char key = waitKey(0);

        if (key == 27) {
            break; // Exit the loop if ESC is pressed
        }

        float q = 0.0f; // Default weight
        int ascii_value = static_cast<int>(key); // Get ASCII value

        if (ascii_value / 100.0f >= 0 && ascii_value / 100.0f <= 1) {
            q = ascii_value / 100.0f; // Calculate q based on ASCII value
        }
        else if (ascii_value > 100) {
            q = 1.0f;
        }

        // Blend the images with the calculated weight
        addWeighted(im1, 1.0f - q, im2, q, 0, im3);
        imshow("Film", im3); // Show blended image

        // Save the blended image if the ASCII value is divisible by 7
        if (ascii_value % 7 == 0) {
            imwrite("keverek.bmp", im3);
        }
    }
}

void showMyImage(Mat& imBig, Mat& im, int& index) {
    im.copyTo(imBig(Rect((index % 6) * (im.cols), (index / 6) * (im.rows),
        im.cols, im.rows)));
    imshow("Ablak", imBig);
    index = (index + 1) % 18;
    waitKey();
}

void lab02() {
    // Betöltjük a színes képet
    Mat im = imread("eper.jpg", 1);
    if (im.empty()) {
        cerr << "Nem sikerült betölteni a képet!" << endl;
        return;
    }

    // Szétszedjük a kép színcsatornáit
    Mat bgr[3], &b = bgr[0], &g = bgr[1], &r = bgr[2], z(im.rows, im.cols, CV_8UC1, Scalar(0));
    split(im, bgr); // BGR csatornák szétszedése

    // Létrehozunk egy nagyobb képet a mozaikszerû elrendezéshez
    Mat imBig = Mat(im.rows * 3, im.cols * 6, im.type());
    imBig.setTo(Scalar(128, 128, 255)); // Színes háttér

    int index = 0;

    showMyImage(imBig, im, index);

    // 1. Egy-egy színcsatorna nullázása
    Mat result = im.clone();
    merge(vector<Mat>{z, g, r}, result); // Kék csatorna 0
    showMyImage(imBig, result, index);
    merge(vector<Mat>{b, z, r}, result); // Zöld csatorna 0
    showMyImage(imBig, result, index);
    merge(vector<Mat>{b, g, z}, result); // Vörös csatorna 0
    showMyImage(imBig, result, index);

    // 2. Két-két színcsatorna nullázása
    merge(vector<Mat>{z, z, r}, result); // Csak vörös
    showMyImage(imBig, result, index);
    merge(vector<Mat>{z, g, z}, result); // Csak zöld
    showMyImage(imBig, result, index);
    merge(vector<Mat>{b, z, z}, result); // Csak kék
    showMyImage(imBig, result, index);

    // 3. Színcsatornák permutálása
    merge(vector<Mat>{b, g, r}, result);
    showMyImage(imBig, result, index);
    merge(vector<Mat>{r, g, b}, result); // RGB (eredeti BGR, de R és B felcserélve)
    showMyImage(imBig, result, index);
    merge(vector<Mat>{g, b, r}, result); // GBR
    showMyImage(imBig, result, index);
    merge(vector<Mat>{g, r, b}, result); // GRB
    showMyImage(imBig, result, index);
    merge(vector<Mat>{b, r, g}, result); // BRG
    showMyImage(imBig, result, index);
    merge(vector<Mat>{r, b, g}, result); // RBG
    showMyImage(imBig, result, index);

    // 4. Egy színkomponens helyettesítése a saját negatívjával
    Mat ib = ~b, ig = ~g, ir = ~r; // Negatív színcsatornák
    merge(vector<Mat>{ib, g, r}, result); // Negatív kék
    showMyImage(imBig, result, index);
    merge(vector<Mat>{b, ig, r}, result); // Negatív zöld
    showMyImage(imBig, result, index);
    merge(vector<Mat>{b, g, ir}, result); // Negatív vörös
    showMyImage(imBig, result, index);

    // 5. Y komponens helyettesítése a saját negatívjával (YCrCb kódolás)
    Mat ycrcb, ycrcb_split[3];
    cvtColor(im, ycrcb, COLOR_BGR2YCrCb); // BGR -> YCrCb
    split(ycrcb, ycrcb_split); // YCrCb színcsatornák szétválasztása
    ycrcb_split[0] = ~ycrcb_split[0]; // Y csatorna negatívja
    merge(ycrcb_split, 3, ycrcb); // YCrCb csatornák összeállítása
    cvtColor(ycrcb, result, COLOR_YCrCb2BGR); // YCrCb -> BGR
    showMyImage(imBig, result, index);

    // 6. Az eredeti kép negatívja
    Mat im_neg = ~im;
    showMyImage(imBig, im_neg, index);
}

void lab03() {
    // load image
    Mat imBe = imread("eper.jpg", 1);
    if (imBe.empty()) {
        cerr << "Could not load the image!" << endl;
        return;
    }

    // show original image
    imshow("original image", imBe);
    waitKey(0);

    // 1.
    // create right shift mask
    float numbers[9] = { 0,0,0,1,0,0,0,0,0 };

    Mat mask = Mat(3, 3, CV_32FC1, numbers);

    Mat imKi = imBe.clone();

    // the image is shifted to the right in several steps
    for (int i = 0; i < 10; ++i) {
        // filter the image with the mask
        filter2D(imKi, imKi, -1, mask);

        // show the shifted image
        imshow("shifted image", imKi);
        waitKey(0);
    }

    // 2.
    // create blur mask
    float numbersForBlur[9] = { 0.1,0.1,0.1,0.1,0.2,0.1,0.1,0.1,0.1 };

    Mat blurMask = Mat(3, 3, CV_32FC1, numbersForBlur);

    imKi = imBe.clone();

    // the image is blurred in several steps
    for (int i = 0; i < 10; ++i) {
        // filter the image with the mask
        filter2D(imKi, imKi, -1, blurMask);

        // show the blurred image
        imshow("blurred image", imKi);
        waitKey(0);
    }

    // 3.
    imKi = imBe.clone();

    float k = 0;
    // the image is sharpened in several steps
    for (int i = 0; i < 10; ++i) {
        k += 0.1;
        float numbersForSharpening[9] = { 0, -k / 4, 0, -k / 4, 1 + k, -k / 4, 0, -k / 4, 0 };
        Mat sharpeningMask = Mat(3, 3, CV_32FC1, numbersForSharpening);
        filter2D(imKi, imKi, -1, sharpeningMask);

        // show the sharpened image
        imshow("sharpened image", imKi);
        waitKey(0);
    }

    // 4.1
    imKi = imBe.clone();

    k = 1;
    for (int i = 0; i < 10; i++) {
        k += 2;
        // blur
        blur(imKi, imKi, Size(k, k));

        imshow("blurred image 2", imKi);
        waitKey(0);
    }

    // 4.2
    imKi = imBe.clone();

    k = 1;
    for (int i = 0; i < 10; i++) {
        k += 2;

        // Gaussian blur
        GaussianBlur(imKi, imKi, Size(k, k), 1);

        imshow("gaussian blurred image", imKi);
        waitKey(0);
    }

    // 5.
    imKi = imBe.clone();

    for (int db = 0; db < 10; ++db) {
        line(imKi, Point(rand() % imKi.cols, rand() % imKi.rows),
            Point(rand() % imKi.cols, rand() % imKi.rows),
            Scalar(0, 0, 0, 0), 1 + db % 2);

        imshow("median blurred image", imKi);
        waitKey(0);
    }

    for (int i = 1; i < 20; i+=2) {
        medianBlur(imKi, imKi, i);

        imshow("median blurred image", imKi);
        waitKey(0);
    }

    // 6.
    imBe = imread("amoba.png", 1);
    if (imBe.empty()) {
        cerr << "Could not load the image!" << endl;
        return;
    }

    resize(imBe, imBe, Size(), 0.25, 0.25);

    // size of the median filter
    int filterSize = 21; 

    imKi = imBe.clone();

    for (int i = 0; i < 80; i++) {
        // Apply the median filter with the given filter size
        medianBlur(imKi, imKi, filterSize);

        // Display the filtered image
        imshow("amoba image", imKi);

        // Wait 50 ms after each step to visually follow the transformation
        waitKey(0);
    }
}

void lab04a() {
    // 1.
    // Load the image as grayscale
    Mat imBe = imread("forest.png", 0);
    if (imBe.empty()) {
        cerr << "Could not load the image!" << endl;
        return;
    }
    resize(imBe, imBe, Size(), 0.25, 0.25);

    // Define the four masks
    float MvpData[9] = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };  // Vertical positive mask
    float MvnData[9] = { 1, 0, -1, 1, 0, -1, 1, 0, -1 };  // Vertical negative mask
    float MfpData[9] = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };  // Horizontal positive mask
    float MfnData[9] = { 1, 1, 1, 0, 0, 0, -1, -1, -1 };  // Horizontal negative mask

    Mat Mvp = Mat(3, 3, CV_32FC1, MvpData);
    Mat Mvn = Mat(3, 3, CV_32FC1, MvnData);
    Mat Mfp = Mat(3, 3, CV_32FC1, MfpData);
    Mat Mfn = Mat(3, 3, CV_32FC1, MfnData);

    // Apply the filters to the image
    Mat Gvp, Gvn, Gfp, Gfn;
    filter2D(imBe, Gvp, -1, Mvp);
    filter2D(imBe, Gvn, -1, Mvn);
    filter2D(imBe, Gfp, -1, Mfp);
    filter2D(imBe, Gfn, -1, Mfn);

    // Show the gradient images
    imshow("Gradient Mvp", Gvp);
    imshow("Gradient Mvn", Gvn);
    imshow("Gradient Mfp", Gfp);
    imshow("Gradient Mfn", Gfn);
    waitKey(0);

    // Combine the gradients
    Mat verticalEdgesGradImg = Gvp + Gvn;
    Mat horizontalEdgesGradImg = Gfp + Gfn;

    // Show the combined gradient images
    imshow("Vertical Edges", verticalEdgesGradImg);
    imshow("Horizontal Edges", horizontalEdgesGradImg);
    waitKey(0);

    // Combine both to get all edges
    Mat allEdgesImg = verticalEdgesGradImg + horizontalEdgesGradImg;

    // Show the final image with all edges
    imshow("All Edges", allEdgesImg);
    waitKey(0);

    // 2.
    Mat thinEdgesImg;

    threshold(allEdgesImg, thinEdgesImg, 110, 255, THRESH_BINARY);

    imshow("Thin Edges", thinEdgesImg);
    waitKey(0);

    // 3.
    Mat edges;
    int lowThreshold = 50;
    int highThreshold = 150;

    Canny(imBe, edges, lowThreshold, highThreshold);

    imshow("Canny Edges", edges);
    waitKey(0);
}


void setChannel(Mat im, int x, int y, int c, uchar v) {
    im.data[y * im.step[0] + x * im.step[1] + c] = v;
}
uchar getChannel(Mat im, int x, int y, int c) {
    return im.data[y * im.step[0] + x * im.step[1] + c];
}

const int Ng = 256;
void drawHist(Mat im) {
    // erre a képre rajzoljuk a hisztogramot
    Mat imHist(Size(3 * Ng, Ng * im.channels()), im.type());
    for (int ch = 0; ch < im.channels(); ++ch) {
        Mat roi = imHist(Rect(0, Ng * ch, 3 * Ng, Ng));
        roi.setTo(Scalar((ch == 0) * (Ng - 1), (ch == 1) * (Ng - 1), (ch == 2) * (Ng - 1), 0));
        // kiszámoljuk a hisztogramot a ch csatornán
        int hist[Ng] = { 0 };
        for (int y = 0; y < im.rows; ++y) {
            for (int x = 0; x < im.cols; ++x)
                hist[getChannel(im, x, y, ch)]++;
        }
        // megkeressük a leggyakoribb intenzitás számosságát
        int maxCol = 0;
        for (int i = 0; i < Ng; ++i) {
            if (hist[i] > maxCol) {
                maxCol = hist[i];
            }
        }
        // megrajzoljuk a hisztogramot a ch csatornán, fentrõl lógó oszlopokkal
        for (int i = 0; i < Ng; ++i) {
            // hármasszabály
            int colHeight = round(250 * hist[i] / maxCol);
            if (colHeight > 0) {
                roi = imHist(Rect(3 * i, Ng * ch, 3, colHeight));
                roi.setTo(Scalar(i, i, i, 0));
            }
        }
        // tükrözzük a kirajzolt hisztogramot, hogy álló oszlopaink legyenek
        roi = imHist(Rect(0, Ng * ch, 3 * Ng, Ng));
        flip(roi, roi, 0);
    }
    resize(imHist, imHist, Size(), 0.5, 0.5);
    imshow("Histogram", imHist);
    waitKey();
}

void drawHistBuiltIn(const Mat& im) {
    vector<Mat> bgr_planes;
    split(im, bgr_planes); // Split the image into BGR channels

    int histSize = 256; // Number of bins
    float range[] = { 0, 256 }; // Range of values
    const float* histRange = { range };

    Mat b_hist, g_hist, r_hist;
    calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, true, false); // B histogram
    calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, true, false); // G histogram
    calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, true, false); // R histogram

    // Normalize the histograms
    normalize(b_hist, b_hist, 0, 255, NORM_MINMAX);
    normalize(g_hist, g_hist, 0, 255, NORM_MINMAX);
    normalize(r_hist, r_hist, 0, 255, NORM_MINMAX);

    int histWidth = 512;
    int histHeight = 400;
    int binWidth = cvRound((double)histWidth / histSize);

    Mat histImage(histHeight, histWidth, CV_8UC3, Scalar(255, 255, 255));

    // Draw each histogram channel
    for (int i = 1; i < histSize; i++) {
        line(histImage,
            Point(binWidth * (i - 1), histHeight - cvRound(b_hist.at<float>(i - 1))),
            Point(binWidth * (i), histHeight - cvRound(b_hist.at<float>(i))),
            Scalar(255, 0, 0), 2, 8, 0); // Blue channel

        line(histImage,
            Point(binWidth * (i - 1), histHeight - cvRound(g_hist.at<float>(i - 1))),
            Point(binWidth * (i), histHeight - cvRound(g_hist.at<float>(i))),
            Scalar(0, 255, 0), 2, 8, 0); // Green channel

        line(histImage,
            Point(binWidth * (i - 1), histHeight - cvRound(r_hist.at<float>(i - 1))),
            Point(binWidth * (i), histHeight - cvRound(r_hist.at<float>(i))),
            Scalar(0, 0, 255), 2, 8, 0); // Red channel
    }

    imshow("Histogram", histImage);
    waitKey(0);
}

void equalizeHistogram(Mat im) {
    if (!(im.channels() % 2)) {
        return;
    }
    // ha színes a kép, átalakítjuk YCrCb kódolásra
    if (im.channels() == 3) {
        cvtColor(im, im, COLOR_BGR2YCrCb); // BGR -> YCrCb
    }
    // most mindenképpen a 0-ás színcsatornát kell nekünk kiegyenlíteni
    // kiszámoljuk a hisztogramot a 0-ás csatornán
    int H[Ng] = { 0 };
    for (int x = 0; x < im.cols; ++x) {
        for (int y = 0; y < im.rows; ++y) {
            H[*im.ptr(y, x)]++;
        }
    }
    // kiszámoljuk az új színeket
    int uj[Ng];
    int sum = 0;
    for (int n = 0; n < Ng; ++n) {
        uj[n] = (sum + H[n] / 2) / (im.cols * im.rows / Ng);
        if (uj[n] > Ng - 1) uj[n] = Ng - 1;
        sum += H[n];
    }
    // átfestjük a képet az új színekkel, a 0-ás csatornán
    for (int x = 0; x < im.cols; ++x) {
        for (int y = 0; y < im.rows; ++y) {
            im.ptr(y, x)[0] = uj[*im.ptr(y, x)];
        }
    }
    // ha színes a kép, visszaalakítjuk BGR kódolásra
    if (im.channels() == 3) {
        cvtColor(im, im, COLOR_YCrCb2BGR);
    }
}

void equalizeHistogramBuiltIn(Mat& im) {
    if (im.channels() == 3) {
        // Convert to YCrCb color space
        Mat ycrcb;
        cvtColor(im, ycrcb, COLOR_BGR2YCrCb);

        // Split channels
        vector<Mat> channels;
        split(ycrcb, channels);

        // Equalize the Y channel using OpenCV's built-in function
        equalizeHist(channels[0], channels[0]);

        // Merge channels back
        merge(channels, ycrcb);

        // Convert back to BGR
        cvtColor(ycrcb, im, COLOR_YCrCb2BGR);
    }
    else {
        // For single-channel image, use OpenCV's built-in equalizeHist
        equalizeHist(im, im);
    }
}

void lab04b() {
    // 1.
    // Load the image as grayscale
    Mat imBe = imread("muzeum.jpg", 1);
    if (imBe.empty()) {
        cerr << "Could not load the image!" << endl;
        return;
    }
    resize(imBe, imBe, Size(), 0.5, 0.5);

    imshow("original image", imBe);

    drawHist(imBe);

    equalizeHistogram(imBe);

    imshow("Equalized histogram", imBe);

    drawHist(imBe);

    // 2.
    imBe = imread("muzeum.jpg", 1);
    resize(imBe, imBe, Size(), 0.5, 0.5);
    imshow("original image", imBe);

    drawHistBuiltIn(imBe);

    equalizeHistogramBuiltIn(imBe);

    imshow("Equalized histogram built in functions", imBe);

    drawHistBuiltIn(imBe);
}

void lab05() {
    Mat imBe = imread("pityoka.png", 0);
    if (imBe.empty()) {
        cerr << "Could not load image!" << endl;
        return;
    }

    resize(imBe, imBe, Size(), 0.5, 0.5);

    // Display the original image
    imshow("Original Image", imBe);

    // 1.
    // create the structuring element
    Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));

    // erode 10x then dilate 10x
    Mat eroded, eroded_dilated;
    erode(imBe, eroded, element, Point(-1, -1), 10);
    dilate(eroded, eroded_dilated, element, Point(-1, -1), 10);

    imshow("Eroded Dilated", eroded_dilated);

    // dilate 10x then erode 10x
    Mat dilated, dilated_eroded;
    dilate(imBe, dilated, element, Point(-1, -1), 10);
    erode(dilated, dilated_eroded, element, Point(-1, -1), 10);

    imshow("Dilated Eroded", dilated_eroded);

    waitKey(0);

    // 2.
    for (int size = 3; size < 21; size+=6)
    {
        // create the structuring element
        Mat element = getStructuringElement(MORPH_ELLIPSE, Size(size, size));

        Mat eroded, dilated;

        erode(imBe, eroded, element);
        imshow("Eroded Image - Size: " + to_string(size), eroded);

        dilate(imBe, dilated, element);
        imshow("Dilated Image - Size: " + to_string(size), dilated);

        waitKey(0);
    }

    // 3.
    imBe = imread("bond.jpg", 1);
    if (imBe.empty()) {
        cerr << "Could not load image!" << endl;
        return;
    }

    resize(imBe, imBe, Size(), 0.5, 0.5);

    Mat imageWithBlackLines = imBe.clone();

    // draw black lines
    line(imageWithBlackLines, Point(50, 50), Point(300, 300), Scalar(0, 0, 0), 5);
    line(imageWithBlackLines, Point(300, 50), Point(50, 300), Scalar(0, 0, 0), 5);

    imshow("Image with black lines", imageWithBlackLines);

    for (int size = 3; size < 21; size+=6)
    {
        Mat element = getStructuringElement(MORPH_RECT, Size(size, size));
        
        Mat dilatedImg;
        dilate(imageWithBlackLines, dilatedImg, element);

        imshow("Dilated image with black lines - Size: " + to_string(size), dilatedImg);

        waitKey(0);
    }

    Mat imageWithWhiteLines = imBe.clone();

    line(imageWithWhiteLines, Point(50, 50), Point(300, 300), Scalar(255, 255, 255), 5);
    line(imageWithWhiteLines, Point(300, 50), Point(50, 300), Scalar(255, 255, 255), 5);

    imshow("Image with white lines", imageWithWhiteLines);

    for (int size = 3; size < 21; size+=6)
    {
        Mat element = getStructuringElement(MORPH_RECT, Size(size, size));

        Mat erodedImg;
        erode(imageWithWhiteLines, erodedImg, element);

        imshow("Eroded image with white lines - Size: " + to_string(size), erodedImg);

        waitKey(0);
    }

    // 4.
    for (int size = 3; size < 21; size+=6)
    {
        Mat element = getStructuringElement(MORPH_ELLIPSE, Size(size, size));
        
        Mat gradientImg;
        morphologyEx(imBe, gradientImg, MORPH_GRADIENT, element);

        imshow("Morphological gradient - Size: " + to_string(size), gradientImg);

        waitKey(0);
    }

    // 5.
    imBe = imread("kukac.png", 0);
    if (imBe.empty()) {
        cerr << "Could not load image!" << endl;
        return;
    }

    resize(imBe, imBe, Size(), 0.25, 0.25);

    // create gradient image
    int width = imBe.cols;
    int height = imBe.rows;
    Mat gradientImage(height, width, CV_8UC1);

    for (int x = 0; x < gradientImage.cols; ++x) {
        gradientImage(Rect(x, 0, 1, gradientImage.rows)).setTo(Scalar(x * 256 / gradientImage.cols));
    }

    // Create a composite image where worms are slightly brighter than the background
    Mat compositeImage;
    addWeighted(gradientImage, 0.9, imBe, 0.1, 0, compositeImage);

    // Apply tophat transformation
    Mat structuringElement = getStructuringElement(MORPH_RECT, Size(11, 11));
    Mat tophatOutput;
    morphologyEx(compositeImage, tophatOutput, MORPH_TOPHAT, structuringElement);

    // Create a composite image where worms are slightly darker than the background
    Mat darkCompositeImage;
    addWeighted(gradientImage, 0.9, imBe, -0.1, 25, darkCompositeImage);

    // Apply blackhat transformation
    Mat blackhatOutput;
    morphologyEx(darkCompositeImage, blackhatOutput, MORPH_BLACKHAT, structuringElement);
    
    // Threshold the outputs
    Mat tophatThresholded, blackhatThresholded;
    threshold(tophatOutput, tophatThresholded, 10, 255, THRESH_BINARY);
    threshold(blackhatOutput, blackhatThresholded, 10, 255, THRESH_BINARY);

    // Show results
    imshow("Original Worms Image", imBe);
    imshow("Gradient Image", gradientImage);
    imshow("Composite Image", compositeImage);
    imshow("Dark Composite Image", darkCompositeImage);
    imshow("Tophat", tophatOutput);
    imshow("Blackhat", blackhatOutput);
    imshow("Tophat Thresholded", tophatThresholded);
    imshow("Blackhat Thresholded", blackhatThresholded);

    waitKey(0);
    return;
}

int getGray(Mat& image, int x, int y) {
    return image.at<uchar>(y, x);
}

void setBlack(Mat& image, int x, int y) {
    image.at<uchar>(y, x) = 0;
}

void golay() {
    static const int Golay[72] = {
   0, 0, 0,
   -1, 1,-1,
   1, 1, 1,
   -1, 0, 0,
   1, 1, 0,
   -1, 1,-1,
   1,-1, 0,
   1, 1, 0,
   1,-1, 0,
   -1, 1,-1,
   1, 1, 0,
   -1, 0, 0,
   ///
   1, 1, 1,
   -1, 1,-1,
   0, 0, 0,
   -1, 1,-1,
   0, 1, 1,
   0, 0,-1,
   0,-1, 1,
   0, 1, 1,
   0,-1, 1,
   0, 0,-1,
   0, 1, 1,
   -1, 1,-1,
    };

    Mat imO = imread("hit_or_miss.png", 0);
    if (imO.empty()) {
        cerr << "Could not load image!" << endl;
        return;
    }

    resize(imO, imO, Size(), 0.5, 0.5);

    Mat imP = imO.clone();
    imshow("golay", imO);

    int count;
    do {
        count = 0;
        for (int l = 0; l < 8; l++) {
            for (int x = 1; x < imO.cols - 1; x++) {
                for (int y = 1; y < imO.rows - 1; y++) {
                    if (getGray(imO, x, y) > 0) {
                        bool erase = true;
                        int index = 9 * l;
                        for (int j = y - 1; j <= y + 1; j++) {
                            for (int i = x - 1; i <= x + 1; i++) {
                                if (Golay[index] == 1 && getGray(imO, i, j) == 0 ||
                                    Golay[index] == 0 && getGray(imO, i, j) > 0) {
                                    erase = false;
                                }
                                index++;
                            }
                        }
                        if (erase) {
                            setBlack(imP, x, y);
                            count++;
                        }
                    }
                }
            }
            imO = imP.clone();
        }
        imshow("Ablak", imP);
        waitKey(100);
    } while (count > 0);
    waitKey(0);
}

void lab06() {
    // 1.
    // Read image
    Mat imBe = imread("amoba.png", 0);
    if (imBe.empty()) {
        cerr << "Could not load image." << endl;
        return;
    }

    resize(imBe, imBe, Size(), 0.7, 0.7);

    imshow("original amoba image", imBe);
    waitKey(0);

    // Perform distanceTransform
    Mat distanceImage;
    distanceTransform(imBe, distanceImage, DIST_L2, 3, CV_32F);

    // Convert 32 bit image to 8 bit
    distanceImage.convertTo(distanceImage, CV_8U, 5, 0);

    // initial distance image
    imshow("distance image", distanceImage);
    waitKey(0);

    // Create 5x5 structuring element
    Mat structuringElement = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));

    Mat dilatedImage;
    for (int size = 3; size < 500; size += 6)
    {   
        dilate(distanceImage, dilatedImage, structuringElement);

        dilatedImage.copyTo(distanceImage, distanceImage);
    }

    imshow("final distance image", distanceImage);
    waitKey(0);

    // 2.
    // Read image
    golay();

    // Color Calibration
    // 3.
    imBe = imread("forest.png", 1);
    if (imBe.empty()) {
        cerr << "Could not load image." << endl;
        return;
    }

    resize(imBe, imBe, Size(), 0.5, 0.5);

    // Initialize LUT (1 row, 256 columns, 3 channels for RGB)
    Mat lut(1, 256, CV_8UC3);
    float brightness = 0;
    float contrast = 2;
    float gamma = 2;

    for (int i = 0; i < 255; i++) 
    {
        // Normalize the pixel value between 0 and 1
        float normalizedValue = i / 255.0;

        // Apply gamma correction
        float gammaCorrectedValue = pow(normalizedValue, 1.0 / gamma);
    
        // Apply contrast adjustment
        float contrastAdjustedValue = contrast * (gammaCorrectedValue - 0.5) + 0.5;
    
        // Apply brightness adjustment
        float brightnessAdjustedValue = brightness + contrastAdjustedValue;
    
        // Clamp the value between 0 and 1
        float clampedValue = min(max(brightnessAdjustedValue, 0.0f), 1.0f);

        // Scale the value back to the range [0, 255]
        int finalValue = static_cast<int>(clampedValue * 255);

        // Set the same value for all three channels (R, G, B)
        lut.at<cv::Vec3b>(0, i) = cv::Vec3b(finalValue, finalValue, finalValue);
    }

    Mat outputImage;
    LUT(imBe, lut, outputImage);

    // Display the results
    imshow("Original image", imBe);
    imshow("Adjusted image", outputImage);
    waitKey(0);

    // 4.
    float b = 0; // brightness
    float c = 2; // 
    float s = 2;
    float t = (1.0 - c) / 2.0;
    float sr = (1 - s) * 0.3086;// 0.3086 or 0.2125
    float sg = (1 - s) * 0.6094;// 0.6094 or 0.7154
    float sb = (1 - s) * 0.0820;// 0.0820 or 0.0721

    float customMatrix[4][4] = {
         {c * (sr + s), c * (sr), c * (sr), 0},
         {c * (sg), c * (sg + s), c * (sg), 0},
         {c * (sb), c * (sb), c * (sb + s), 0},
         {t + b, t + b, t + b, 1}
    };

    Mat rgbaImage;
    imBe.convertTo(rgbaImage, CV_32FC4, 1 / 255.);
    cvtColor(rgbaImage, rgbaImage, COLOR_BGR2BGRA);

    rgbaImage = rgbaImage.reshape(1, imBe.cols * imBe.rows);
    rgbaImage *= Mat(4, 4, CV_32FC1, customMatrix);
    rgbaImage = rgbaImage.reshape(4, imBe.rows);

    Mat rgbImage;
    cvtColor(rgbaImage, rgbImage, COLOR_BGRA2BGR);
    rgbImage.convertTo(rgbImage, CV_8UC3, 255);

    imshow("Transformed image", rgbImage);
    waitKey(0);
}

void lab07() {
    // Load brain image as grayscale
    Mat im = imread("brain.bmp", 0);
    if (im.empty()) {
        cerr << "Could not load image." << endl;
        return;
    }

    resize(im, im, Size(), 1.5, 1.5);

    // Show input brain image
    imshow("Input brain image", im);

    float m_values[] = {1.5f, 2.0f, 3.0f};

    for (int m_idx = 0; m_idx < 3; m_idx++)
    {
        // Constants
        const int Ng = 256;
        const int c = 3;
        const float m = m_values[m_idx];
        const float mm = -1.0f / (m - 1.0f);
        const float eps = 0.00000001;

        // Initialize arrays
        float u[c][Ng];
        float v[c];
        float d2[c][Ng];
        int H[Ng] = { 0 };

        // Compute the histogram of the input image
        for (int row = 0; row < im.rows; row++) {
            for (int col = 0; col < im.cols; col++) {
                H[im.at<uchar>(row, col)]++;
            }
        }

        // Initialize the class prototypes v[i] to distinct values
        for (int i = 0; i < c; ++i) {
            v[i] = 255.0f * (i + 1.0f) / (2.0f * c);
        }

        // A konvergenciához húsz ciklus elég lesz
        for (int ciklus = 0; ciklus < 20; ++ciklus) {
            // A partíciós matrix újraszámítása. Beiktattunk egy védelmet arra az esetre amikor d2[][]
            // valamely értéke zéró, mert ilyenkor zéróval kellene osztanunk
            for (int l = 0; l < Ng; ++l) {
                for (int i = 0; i < c; ++i) {
                    d2[i][l] = (l - v[i]) * (l - v[i]);
                }
                int winner = 0;
                for (int i = 0; i < c; ++i) {
                    if (d2[winner][l] > d2[i][l]) {
                        winner = i;
                    }
                }
                if (d2[winner][l] < eps) {
                    for (int i = 0; i < c; ++i) {
                        u[i][l] = 0.0f;
                    }
                    u[winner][l] = 1.0f;
                }
                else {
                    float sum = 0;
                    for (int i = 0; i < c; ++i) {
                        u[i][l] = pow(d2[i][l], mm);
                        sum += u[i][l];
                    }
                    for (int i = 0; i < c; ++i) {
                        u[i][l] /= sum;
                    }
                }
            }
            // az osztályprototípusok újraszámolása
            for (int i = 0; i < c; ++i) {
                float sumUp = 0.0f;
                float sumDn = 0.0f;
                for (int l = 0; l < Ng; ++l) {
                    sumUp += H[l] * pow(u[i][l], m) * l;
                    sumDn += H[l] * pow(u[i][l], m);
                }
                v[i] = sumUp / sumDn;
            }
        }

        // Create a lookup table (LUT) for classifying the image
        Mat lut(1, 256, CV_8U);
        for (int l = 0; l < Ng; ++l) {
            int winner = 0;
            for (int i = 1; i < c; ++i) {
                if (u[i][l] > u[winner][l]) {
                    winner = i;
                }
            }
            lut.at<uchar>(0, l) = round(v[winner]);
        }
        // Apply the LUT to the image
        cv::LUT(im, lut, im);

        // Display the segmented image for the current m
        string window_name = "Segmented Image (m = " + to_string(m) + ")";
        imshow(window_name, im);

        // Visualize the fuzzy membership functions
        Mat imF(Size(768, 400), CV_8UC3, Scalar(0, 0, 0, 0));
        for (int i = 0; i < c; ++i) {
            for (int l = 0; l < Ng; ++l) {
                circle(imF, Point(1 + 3 * l, round(400.0 * (1.0f - u[i][l]))), 2,
                    Scalar(255 * (i == 0), 255 * (i == 1), 255 * (i == 2), 0));
            }
        }
        // Display the fuzzy membership functions
        string plot_name = "Fuzzy Membership Functions (m = " + to_string(m) + ")";
        imshow(plot_name, imF);
        waitKey();
    }
}

void lab08() {
    // A
    Mat imBe = imread("hod.jpg", 1);
    if (imBe.empty()) {
        cerr << "Could not load image." << endl;
        return;
    }

    // given circle radius
    const int R = 89;

    // search for the best color channel (circles are most visible)
    Mat channels[3];
    split(imBe, channels);

    Mat imBlue = channels[0]; // we should not use this one in our case
    Mat imGreen = channels[1]; // we can use this
    Mat imRed = channels[2]; // or this
    //imshow("Blue channel", imBlue);
    //imshow("Green channel", imGreen);
    //imshow("Red channel", imRed);
    //waitKey(0);

    // We will work with the blue channel (circles most visible)
    // Run Canny edge detection
    Mat edges;
    Canny(imGreen, edges, 40, 100);

    Mat hough_acc = Mat::zeros(imBe.size(), CV_32SC1); // Hough space accumulator

    // save coordinates of circle with radius R into an array
    vector<Point> circPoint;
    for (int theta = 0; theta < 360; ++theta) {
        int x = cvRound(R * cos(theta * CV_PI / 180.0));
        int y = cvRound(R * sin(theta * CV_PI / 180.0));
        circPoint.push_back(Point(x, y));
    }

    for (int y = 0; y < edges.rows; y++) {
        for (int x = 0; x < edges.cols; x++) {
            if (edges.at<uchar>(y, x) > 0) {
                for (size_t i = 0; i < circPoint.size(); ++i) {
                    int a = x + circPoint[i].x;
                    int b = y + circPoint[i].y;
                    if (a >= 0 && a < hough_acc.cols && b >= 0 && b < hough_acc.rows) {
                        hough_acc.at<int>(b, a)++;
                    }
                }
            }
        }
    }

    Mat hough_display;
    normalize(hough_acc, hough_display, 0, 255, NORM_MINMAX, CV_8UC1);

    imshow("Circle Center Accumulator", hough_display);
    waitKey(0);

    vector<Point> circle_centers;
    int maxVotes = 100;
    for (int y = 0; y < hough_acc.rows; y++) {
        for (int x = 0; x < hough_acc.cols; x++) {
            if (hough_acc.at<int>(y, x) > maxVotes) {
                circle_centers.push_back(Point(x, y));

                circle(hough_acc, Point(x, y), R / 2, Scalar(0), -1);
            }
        }
    }

    for (size_t i = 0; i < circle_centers.size(); i++) {
        circle(imBe, circle_centers[i], R, Scalar(0, 0, 255), 2);
    }

    imshow("Detected Circle", imBe);
    waitKey(0);

    // B
    imBe = imread("hod.jpg", 1);
    if (imBe.empty()) {
        cerr << "Image not found!" << endl;
        return;
    }

    Mat gray;
    cvtColor(imBe, gray, COLOR_BGR2GRAY);

    medianBlur(gray, gray, 3);

    vector<Vec3f> circles;
    HoughCircles(gray, circles, HOUGH_GRADIENT, 1,
        gray.rows / 16,
        100, 30, R-10, R+10);

    for (size_t i = 0; i < circles.size(); i++)
    {
        Vec3i c = circles[i];
        Point center = Point(c[0], c[1]);
        int radius = c[2];
        circle(imBe, center, radius, Scalar(0, 0, 255), 3, LINE_AA);
    }

    imshow("detected circles with HoughCircles", imBe);
    waitKey(0);
}

// Function to get the color of a pixel at (x, y) in a color image.
// The channels are ordered as Blue, Green, Red.
void getColor(Mat im, int x, int y, uchar& blue, uchar& green, uchar& red) {
    Vec3b color = im.at<Vec3b>(y, x); // Access the pixel at (y, x)
    blue = color[0];   // Blue channel
    green = color[1];  // Green channel
    red = color[2];    // Red channel
}

// Function to set the color of a pixel at (x, y) in a color image.
// The channels are ordered as Blue, Green, Red.
void setColor(Mat im, int x, int y, uchar blue, uchar green, uchar red) {
    im.at<Vec3b>(y, x) = Vec3b(blue, green, red); // Set pixel value at (y, x)
}

// Function to get the grayscale value of a pixel at (x, y) in a grayscale image.
uchar getGray(const Mat& im, int x, int y) {
    return im.at<uchar>(y, x); // Return the pixel value at (y, x)
}

// Function to set the grayscale value of a pixel at (x, y) in a grayscale image.
void setGray(Mat& im, int x, int y, uchar v) {
    im.at<uchar>(y, x) = v; // Set the pixel value at (y, x)
}

// Comparison function for the quicksort algorithm.
// Returns 1 if p1 > p2, -1 if p1 < p2, 0 if p1 == p2.
int compare(const void* p1, const void* p2) {
    int a = *(const int*)p1;
    int b = *(const int*)p2;
    if (a > b) return 1;
    if (a < b) return -1;
    return 0;
}

void lab09() {
	const uchar bits[8] = { 1, 2, 4, 8, 16, 32, 64, 128 };

	// a kifelé folyás irányainak x és y komponense
	const int dx[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
	const int dy[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };

	// betöltjük a szegmentálandó színes képet
	Mat imColor = imread("3.jpg", 1);

	// készítünk belõle egy szürke verziót
	Mat imO(imColor.cols, imColor.rows, CV_8UC1);
	cvtColor(imColor, imO, COLOR_BGR2GRAY);

	imshow("Szines", imColor);
	waitKey();

	imshow("Szurke", imO);
	waitKey();

	// Ez a kép tárolja majd a kiszámított gradiens értéket minden képpontban
	Mat imG = imO.clone();

	// Ez a kép tárolja majd minden képpont szomszédságában található legkisebb gradiens értéket
	Mat imE = imO.clone();

	// Ez a kép tárolja a kifele folyás irányát minden képpontban
	Mat imKi = imO.clone();

	// Ez a kép tárolja a befele folyás irányait (bitenként) minden képpontban
	Mat imBe = imO.clone();

	// A szegmentált kép lesz (vízgyûjtõk átlagolt színével)
	Mat imSegm = imColor.clone();

	// A szegmentált kép lesz (vízgyûjtõk medián színével)
	Mat imSegmMed = imColor.clone();

	// Bináris kép mely megmutatja, hogy melyik képpontokban állítottuk már be a kifele folyást
	Mat imMap = imO.clone();

	// Gradiensek számításához használt 16 bites elõjeles kódolású kép
	Mat imL(imColor.cols, imColor.rows, CV_16SC1);

	// felbontjuk a színes képet színcsatornáira
	vector<Mat> imColors;
	split(imColor, imColors);

	// A bemeneti kép három színcsatornáját fogjuk itt tárolni, ezekbõl számoljuk a
	// gradienseket
	Mat imRed = imColors[2];
	Mat imGreen = imColors[1];
	Mat imBlue = imColors[0];

	// Ezen a képen összegezzük a három színcsatorna gradienseit
	Mat imSum = imO.clone();
	imSum.setTo(Scalar(0));

	// kék színcsatorna gradienseit adjuk hozzá az imSum-hoz
	Sobel(imBlue, imL, imL.depth(), 1, 0);
	convertScaleAbs(imL, imE);
	Sobel(imBlue, imL, imL.depth(), 0, 1);
	convertScaleAbs(imL, imG);
	add(imE, imG, imG);
	addWeighted(imSum, 1, imG, 0.33333, 0, imSum);

	// zõd színcsatorna gradienseit adjuk hozzá az imSum-hoz
	Sobel(imGreen, imL, imL.depth(), 1, 0);
	convertScaleAbs(imL, imE);
	Sobel(imGreen, imL, imL.depth(), 0, 1);
	convertScaleAbs(imL, imG);
	add(imE, imG, imG);
	addWeighted(imSum, 1, imG, 0.33333, 0, imSum);

	// vörös színcsatorna gradienseit adjuk hozzá az imSum-hoz
	Sobel(imRed, imL, imL.depth(), 1, 0);
	convertScaleAbs(imL, imE);
	Sobel(imRed, imL, imL.depth(), 0, 1);
	convertScaleAbs(imL, imG);
	add(imE, imG, imG);
	addWeighted(imSum, 1, imG, 0.33333, 0, imG);

	// Az összesített gradiens az imG képbe került
	//Elõfeldolgozási lépés, amelyek közül csak egyiket használjuk
	// 1 - gradiensek csonkolása
	//cvCmpS(imG, 32, imE, CV_CMP_LT);
	//cvSub(imG, imG, imG, imE);
	// 2 - a gradiensek kisimítása egy Gauss-féle aluláteresztõ szûrõvel
	GaussianBlur(imG, imG, Size(9, 9), 0);
	imshow("Gradiens", imG);
	waitKey();

	// Step 0 - inicializálás
	// Erodált gradiensek kiszámítása - a szomszédságban levõ legkisebb gradiensek kiszámítása
	erode(imG, imE, getStructuringElement(MORPH_RECT, Size(3, 3)));

	// A szegmentált képeket inicializáljuk egy szürke árnyalattal, ezeket elvileg mind felül fogja írni az algoritmus
	imSegm.setTo(Scalar(50, 50, 50));
	imSegmMed.setTo(Scalar(150, 150, 150));

	// Egyik pixelnél sincs befelé folyás kezdetben
	imBe.setTo(Scalar(0));

	// Valódi kifelé folyási irányok: 0..7, 8 azt jelenti, hogy az adott pixelnél még nincs eldöntve a kifelé folyás iránya
	imKi.setTo(Scalar(8));

	// Kezdetben sehol nincs még eldöntve a kifelé folyás iránya
	imMap.setTo(Scalar(0));

	// Step 1 - keressük meg és kezeljük le az összes képpontot, ahol a gradiens térkép lejtõs
		// Bejárjuk a képet (x,y)-nal
	for (int x = 0; x < imBe.cols; ++x) {
		for (int y = 0; y < imBe.rows; ++y) {
			int fp = getGray(imG, x, y);
			int q = getGray(imE, x, y);
			// ahol az erodált gradiens kisebb a lokális gradiensnél, ott lejtõs helyen vagyunk
			if (q < fp) {
				// megkeressük, hogy melyik irányba a legmeredekebb a lejtõ
				for (uchar irany = 0; irany < 8; ++irany) {
					// létezik-e a vizsgált koordinátájú szomszéd
					if (x + dx[irany] >= 0 && x + dx[irany] < imBe.cols && y + dy[irany]
						>= 0 &&
						y + dy[irany] < imBe.rows) {
						int fpv = getGray(imG, x + dx[irany], y + dy[irany]);
						// ha az adott irany szerinti szomszéd gradiense annyi mint a
						// minimum gradiens a szomszédságban...
						if (fpv == q) {
							//...akkor beállítjuk a kifelé folyást az adott szomszéd irányába
							setGray(imKi, x, y, irany);
							// bejelöljük, hogy az (x,y) képpontban megvan a kifelé folyás iránya
							setGray(imMap, x, y, 255);
							// kiolvassuk a befelé folyás bitjeit a szomszédban...
							uchar volt = getGray(imBe, x + dx[irany], y + dy[irany]);
							// megmódosítjuk ...
							uchar adunk = bits[irany];
							uchar lesz = volt | adunk;
							// és visszaírjuk
							setGray(imBe, x + dx[irany], y + dy[irany], lesz);
							break;
						}
					}
				}
			}
		}
	}

	// megmutatjuk egy ablakban a lekezelt képpontok térképét és várunk gombnyomásra
	imshow("Lekezelt képpontok térképe", imMap);
	waitKey();

	// Step 2 - fennsíkon levõ pontok lekezelése a gradiens térképen
	// Kell egy FIFO lista amire képpontokat fogunk elhelyezni
	Point* fifo = new Point[imBe.cols * imBe.rows];
	int nextIn = 0;
	int nextOut = 0;

	// Bejárjuk a képet (x,y)-nal
	for (int x = 0; x < imBe.cols; ++x) {
		for (int y = 0; y < imBe.rows; ++y) {
			// olyan képpontot keresünk, ahol már el van döntve a kifelé folyás iránya de
			// van olyan szomszédja, ahol még nincs eldöntve
			int fp = getGray(imG, x, y);
			int pout = getGray(imKi, x, y);
			if (pout == 8) continue;
			// találtunk egy olyan képpontot, ahol a kifelé folyás iránya már el van döntve ...
			int added = 0;
			// ... és vizsgáljuk annak a szomszédjait
			for (uchar irany = 0; irany < 8; ++irany) {
				if (x + dx[irany] >= 0 && x + dx[irany] < imBe.cols && y + dy[irany] >= 0
					&&
					y + dy[irany] < imBe.rows) {
					int fpv = getGray(imG, x + dx[irany], y + dy[irany]);
					int pvout = getGray(imKi, x + dx[irany], y + dy[irany]);
					if (fpv == fp && pvout == 8) {
						// ha ide jutunk, akkor találtunk olyan szomszédot, ahol még
						// nincs eldöntve a kifelé folyás iránya
							// az ilyen (x,y) képpontokat felvesszük a FIFO listára
						if (added == 0) fifo[nextIn++] = Point(x, y);
						added++;
					}
				}
			}
		}
	}

	// amíg ki nem ürül a FIFO lista
	while (nextOut < nextIn) {
		// kiveszünk egy képpontot a listáról
		Point p = fifo[nextOut++];
		int fp = getGray(imG, p.x, p.y);
		// megkeressük az összes olyan szomszédját, ahol még nincs eldöntve a kifolyás iránya
		for (uchar irany = 0; irany < 8; ++irany) {
			if (p.x + dx[irany] >= 0 && p.x + dx[irany] < imBe.cols && p.y + dy[irany] >=
				0 &&
				p.y + dy[irany] < imBe.rows) {
				int fpv = getGray(imG, p.x + dx[irany], p.y + dy[irany]);
				int pvout = getGray(imKi, p.x + dx[irany], p.y + dy[irany]);
				if (fp == fpv && pvout == 8) {
					// bejelöljük a kifelé folyás irányát a szomszédtól felénk
					setGray(imKi, p.x + dx[irany], p.y + dy[irany], (irany + 4) % 8);
					// bejelöljük, hogy a szomszéd képpontban megvan a kifelé folyás
					// iránya
					setGray(imMap, p.x + dx[irany], p.y + dy[irany], 255);
					// bejelöljük a befelé folyás irányát
					setGray(imBe, p.x, p.y, bits[(irany + 4) % 8] | getGray(imBe, p.x,
						p.y));
					// az újonnan bejelölt szomszéd felkerül a listára
					fifo[nextIn++] = Point(p.x + dx[irany], p.y + dy[irany]);
				}
			}
		}
	}

	// megmutatjuk az ablakban a lekezelt képpontok térképét és várunk gombnyomásra
	imshow("Lekezelt képpontok térképe", imMap);
	waitKey();

	// Step 3 - megkeressük a völgyekhez tartozó képpontokat a gradiens térképen
	// Keresünk olyan képpontot, amilyikbõl még nincs bejelölve a kifelé folyás iránya
	// Az ilyen képpontot kinevezzük lokális minimumnak és megkeressük körülötte azon
	// pontokat, amelyiknek még nincs kifelé folyása, ezekbõl mind a lokális minimum felé
	// fog folyni a víz
		// Szükségünk van egy veremre
	Point* stack = new Point[imBe.cols * imBe.rows];
	int nrStack = 0;
	// Bejárjuk a képet (x,y)-nal
	for (int x = 0; x < imBe.cols; ++x) {
		for (int y = 0; y < imBe.rows; ++y)
		{
			int fp = getGray(imG, x, y);
			int pout = getGray(imKi, x, y);
			// Amelyik képpontban már megvan a kifelé folyás irányam azzal nem kell foglalkozni
			if (pout != 8) continue;
			// pout egy lokális minimumnak lesz kinevezve
			// Megkeressük azokat a szomszédokat, amelyeknek még nincs meg a kifelé folyási irányuk
			for (uchar irany = 0; irany < 8; ++irany) {
				if (x + dx[irany] >= 0 && x + dx[irany] < imBe.cols && y + dy[irany] >= 0 &&
					y + dy[irany] < imBe.rows)
				{
					int fpv = getGray(imG, x + dx[irany], y + dy[irany]);
					int pvout = getGray(imKi, x + dx[irany], y + dy[irany]);
					if (pvout == 8 && fp == fpv)
					{
						// itt találtunk olyan szomszédot
					   // bejelöljük a kifelé folyást a lokális minimum felé
						setGray(imKi, x + dx[irany], y + dy[irany], (irany + 4) % 8);
						setGray(imMap, x + dx[irany], y + dy[irany], 255);
						setGray(imBe, x, y, bits[(irany + 4) % 8] | getGray(imBe, x, y));
						// a szomszéd képpontot betesszük a verembe
						stack[nrStack++] = Point(x + dx[irany], y + dy[irany]);
					}
				}
			}
			// amíg ki nem ürül a verem
			while (nrStack > 0)
			{
				// kiveszünk egy képpontot és megnézzük, hogy a szomszédai között van-e
				// olyan, akinek nincs megjelölve a kifelé folyás iránya
				Point pv = stack[--nrStack];
				int fpv = getGray(imG, pv.x, pv.y);
				int pvout = getGray(imKi, pv.x, pv.y);
				for (uchar irany = 0; irany < 8; ++irany) {
					if (pv.x + dx[irany] >= 0 && pv.x + dx[irany] < imBe.cols && pv.y +
						dy[irany] >= 0 &&
						pv.y + dy[irany] < imBe.rows) {
						// itt találtunk létezõ szomszédot
						int fpvv = getGray(imG, pv.x + dx[irany], pv.y + dy[irany]);
						int pvvout = getGray(imKi, pv.x + dx[irany], pv.y + dy[irany]);
						//if (fpv==fpvv && pvvout==8 && (!(pv.x+dx[pvout]==x && pv.y + dy[pvout] == y)))
						if (fpv == fpvv && pvvout == 8 && (!(pv.x + dx[irany] == x &&
							pv.y + dy[irany] == y))) {
							// itt találtunk olyan szomszédot
						   // bejelöljük a kifelé folyást pout felé
							setGray(imMap, pv.x + dx[irany], pv.y + dy[irany], 255);
							setGray(imKi, pv.x + dx[irany], pv.y + dy[irany], (irany + 4)
								% 8);
							setGray(imBe, pv.x, pv.y, bits[(irany + 4) % 8] |
								getGray(imBe, pv.x, pv.y));
							// a szomszéd képpontot betesszük a verembe
							stack[nrStack++] = Point(pv.x + dx[irany], pv.y + dy[irany]);
						}
					}
				}
			}
		}
	}

	// megmutatjuk az ablakban a lekezelt képpontok térképét és várunk gombnyomásra
	// itt már csak izolált fekete képpontok lesznek a fehér képen, ezek a lokális
	// minimumok
	imshow("Lekezelt képpontok térképe", imMap);
	waitKey();

	// Step 4
	// feltérképezzük a vízgyûjtõ medencéket a lokális minimumokból kiindulva a víz
	// folyásával fordított irányba haladva
		// minden vízgyûjtõ medencében kiszámoljuk az átlagos és a medián színt
		// mindkettõbõl generálunk egy-egy külön kimeneti szegmentált képet
		// ez a puffer a medián számításához kell
	uint* medbuff = new uint[imBe.cols * imBe.rows];
	int label = 0;
	nextIn = 0;
	int spotSum[3];
	// Bejárjuk a képet (x,y)-nal
	for (int x = 0; x < imBe.cols; ++x) for (int y = 0; y < imBe.rows; ++y) {
		// keresünk lokális mimimumot
		int pout = getGray(imKi, x, y);
		if (pout != 8) continue;
		// találtunk lokális mimimumot, betesszük a verembe
		stack[nrStack++] = Point(x, y);
		for (int i = 0; i < 3; ++i) { spotSum[i] = 0; }
		// amíg üres nem lesz a verem
		while (nrStack > 0) {
			// Kiveszünk egy képpontot a verembõl és megnézzük, honnan folyik felénk a
			// víz
				// Ahonnan felénk folyik a víz, azt a képpontot felvesszük az aktuális
				// régióba és
				// betesszük a verembe is.
			Point pv = stack[--nrStack];
			fifo[nextIn++] = pv;
			uchar r, g, b;
			getColor(imColor, pv.x, pv.y, r, g, b);
			spotSum[0] += (int)b;
			spotSum[1] += (int)g;
			spotSum[2] += (int)r;
			uint o = (int)r * 0x10000 + (int)g * 0x100 + (int)b;
			o += (uint)(round((float)r * 0.299 +
				(float)g * 0.587 + (float)b * 0.114) *
				0x1000000);
			medbuff[nextIn - 1] = o;
			// setGray(imLabel, pv.x, pv.y, label);
			int pvin = getGray(imBe, pv.x, pv.y);
			for (uchar irany = 0; irany < 8; ++irany) {
				if ((bits[irany] & pvin) > 0) {
					// setGray(imLabel, pv.x + dx[(irany + 4) % 8], pv.y + dy[(irany + 4) % 8],
					// label);
					stack[nrStack++] = Point(pv.x + dx[(irany + 4) % 8], pv.y +
						dy[(irany + 4) % 8]);
				}
			}
		}
		// a label azt számolja, hogy hány régió van összesen a szegmentált képen
		label++;
		if (nextIn < 2) printf("%d", nextIn);
		for (int i = 0; i < 3; ++i) {
			spotSum[i] = round(spotSum[i] / nextIn);
		}
		// kiszámoljuk a medián színt a quicksort segítségével
		qsort(medbuff, nextIn, sizeof(uint), compare);
		int medR = (medbuff[nextIn / 2] % 0x1000000) / 0x10000;
		int medG = (medbuff[nextIn / 2] % 0x10000) / 0x100;
		int medB = (medbuff[nextIn / 2] % 0x100);
		for (int i = 0; i < nextIn; ++i) //if (getGray(imMask, fifo[i].x, fifo[i].y) > 128)
		{
			// itt festjük ki a régiót az átlagos színnel
			setColor(imSegm, fifo[i].x, fifo[i].y, (uchar)spotSum[2], (uchar)
				spotSum[1], (uchar)spotSum[0]);
			// itt festjük ki a régiót a medián színnel
			setColor(imSegmMed, fifo[i].x, fifo[i].y, (uchar)medR, (uchar)medG,
				(uchar)medB);
		}
		nextIn = 0;
	}
	// memória felszabadítás
	free(fifo);
	free(stack);
	free(medbuff);
	// no more steps
	printf("\nRegions: %d \n", label);
	// megmutatjuk egy ablakban a medián színekkel készített képet
	imshow("Median", imSegmMed);
	// megmutatjuk egy masik ablakban az átlagos színekkel készített képet
	imshow("Atlag", imSegm);
	waitKey();
}

void startVideo(VideoCapture& cap, const string& filename) {
    cap.open(filename);
    if (!cap.isOpened()) {
        cout << "Error when opening video stream or file" << endl;
        return;
    }

    return;
}

// Region growing function
int regionGrowing(Mat im, Point p0, Point& pbf, Point& pja) {
    int count = 0;
    Point* fifo = new Point[0x100000];
    int nextIn = 0;
    int nextOut = 0;
    pbf = p0;
    pja = p0;
    if (getGray(im, p0.x, p0.y) < 128)
        return 0;
    fifo[nextIn++] = p0;
    setGray(im, p0.x, p0.y, 100);
    while (nextIn > nextOut) {
        Point p = fifo[nextOut++];
        ++count;
        if (p.x > 0)
            if (getGray(im, p.x - 1, p.y) > 128)
            {
                fifo[nextIn++] = Point(p.x - 1, p.y);
                setGray(im, p.x - 1, p.y, 100);
                if (pbf.x > p.x - 1) pbf.x = p.x - 1;
            }
        if (p.x < im.cols - 1)
            if (getGray(im, p.x + 1, p.y) > 128)
            {
                fifo[nextIn++] = Point(p.x + 1, p.y);
                setGray(im, p.x + 1, p.y, 100);
                if (pja.x < p.x + 1) pja.x = p.x + 1;
            }
        if (p.y > 0)
            if (getGray(im, p.x, p.y - 1) > 128)
            {
                fifo[nextIn++] = Point(p.x, p.y - 1);
                setGray(im, p.x, p.y - 1, 100);
                if (pbf.y > p.y - 1) pbf.y = p.y - 1;
            }
        if (p.y < im.rows - 1)
            if (getGray(im, p.x, p.y + 1) > 128)
            {
                fifo[nextIn++] = Point(p.x, p.y + 1);
                setGray(im, p.x, p.y + 1, 100);
                if (pja.y < p.y + 1) pja.y = p.y + 1;
            }
    }
    delete[]fifo;
    return count;
}

void lab10() {
    // read video
    /*VideoCapture cap("IMG_6909.MOV");
    if (!cap.isOpened()) {
        cout << "Error when opening video stream or file" << endl;
        return;
    }*/

    // Get the frames per second (FPS) of the video
    // double fps = cap.get(cv::CAP_PROP_FPS);
    // cout << "Frames per second (FPS): " << fps << endl;
    // 
    // the video is 30 fps so the waitKey delay for that is 33
    // but we need to play it in 60 fps so we halve the 33, which is 16

    VideoCapture cap;

    while (1) {
        startVideo(cap, "IMG_6909.MOV");

        Mat frame;

        while (cap.read(frame)) {
            if (frame.empty()) {
                break;
            }

            resize(frame, frame, Size(), 0.1, 0.1);

            // colored playback
            imshow("color video", frame);

            // black and white (grayscale) video
            Mat grayFrame;
            cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
            imshow("black and white video", grayFrame);

            if (waitKey(16) == 32) {
                cap.release();
                destroyAllWindows();
                break;
            }
        }

        startVideo(cap, "IMG_6909.MOV");

        while (cap.read(frame)) {
            if (frame.empty()) {
                break;
            }

            resize(frame, frame, Size(), 0.1, 0.1);

            Mat grayFrame;
            cvtColor(frame, grayFrame, COLOR_BGR2GRAY);

            // Canny edge detection
            Mat edgesVideo;
            Canny(grayFrame, edgesVideo, 100, 200);
            imshow("edges video", edgesVideo);

            if (waitKey(16) == 32) {
                cap.release();
                destroyAllWindows();
                break;
            }
        }

        startVideo(cap, "IMG_6909.MOV");

        while (cap.read(frame)) {
            if (frame.empty()) {
                break;
            }

            resize(frame, frame, Size(), 0.1, 0.1);

            // Median blur
            Mat medianBlurredVideo;
            medianBlur(frame, medianBlurredVideo, 5);
            imshow("median blurred video", medianBlurredVideo);

            if (waitKey(16) == 32) {
                cap.release();
                destroyAllWindows();
                break;
            }
        }

        startVideo(cap, "IMG_6909.MOV");

        while (cap.read(frame)) {
            if (frame.empty()) {
                break;
            }

            resize(frame, frame, Size(), 0.1, 0.1);

            // Gaussian blur
            Mat gaussianBlurredVideo;
            GaussianBlur(frame, gaussianBlurredVideo, Size(15, 15), 0);
            imshow("Gaussian blurred video", gaussianBlurredVideo);

            if (waitKey(16) == 32) {
                cap.release();
                destroyAllWindows();
                break;
            }
        }

        startVideo(cap, "IMG_6909.MOV");

        while (cap.read(frame)) {
            if (frame.empty()) {
                break;
            }

            resize(frame, frame, Size(), 0.1, 0.1);

            // Low pass filter
            float numbersForBlur[9] = { 0.1,0.1,0.1,0.1,0.2,0.1,0.1,0.1,0.1 };
            Mat blurMask = Mat(3, 3, CV_32FC1, numbersForBlur);

            Mat lowPassFilteredImage;
            filter2D(frame, lowPassFilteredImage, -1, blurMask);
            imshow("low pass filtered image", lowPassFilteredImage);

            if (waitKey(16) == 32) {
                cap.release();
                destroyAllWindows();
                break;
            }
        }

        startVideo(cap, "IMG_6909.MOV");

        while (cap.read(frame)) {
            if (frame.empty()) {
                break;
            }

            resize(frame, frame, Size(), 0.1, 0.1);

            Mat grayFrame;
            cvtColor(frame, grayFrame, COLOR_BGR2GRAY);

            // High pass filter (Laplace)
            Mat laplaceFilteredVideo;
            Laplacian(grayFrame, laplaceFilteredVideo, CV_64F);
            imshow("Laplace filtered video", laplaceFilteredVideo);

            if (waitKey(16) == 32) {
                cap.release();
                destroyAllWindows();
                break;
            }
        }

        startVideo(cap, "IMG_6909.MOV");

        while (cap.read(frame)) {
            if (frame.empty()) {
                break;
            }

            resize(frame, frame, Size(), 0.1, 0.1);

            // Histogram equalization
            equalizeHistogram(frame);
            imshow("histogram equalized video", frame);

            // Histogram equalization with built-in function
            equalizeHistogramBuiltIn(frame);
            imshow("histogram equalized video with built-in function", frame);

            if (waitKey(16) == 32) {
                cap.release();
                destroyAllWindows();
                break;
            }
        }

        startVideo(cap, "IMG_6909.MOV");

        while (cap.read(frame)) {
            if (frame.empty()) {
                break;
            }

            resize(frame, frame, Size(), 0.1, 0.1);

            // Resizing
            Mat resizedFrame;
            resize(frame, resizedFrame, Size(frame.cols * 2, frame.rows * 2));
            imshow("resized video", resizedFrame);

            if (waitKey(16) == 32) {
                cap.release();
                destroyAllWindows();
                break;
            }
        }

        startVideo(cap, "IMG_6909.MOV");

        int flag = 1;
        Mat background;
        int frameCount = 0;

        while (cap.read(frame)) {
            if (frame.empty()) {
                break;
            }

            // Motion detection
            resize(frame, frame, Size(frame.cols / 4, frame.rows / 4));

            if (flag) {
                background = frame.clone();
                flag = 0;
            }

            // Optimizing video speed
            //if (frameCount++ % 2 != 0) continue;

            // Difference between background and current frame
            Mat diffFrame = (background - frame) + (frame - background);

            // Split the frame into its channels, and calculate the sum of the channels
            Mat channels[3];
            split(diffFrame, channels);
            Mat combinedFrame = channels[0] + channels[1] + channels[2];

            // Binarization using compare
            Mat binaryFrame;
            compare(combinedFrame, 170, binaryFrame, CMP_GE);

            // Remove a great portion of white spots using erosion
            Mat structuringElement = getStructuringElement(MORPH_RECT, Size(9, 9));
            Mat erodedFrame;
            erode(binaryFrame, erodedFrame, structuringElement);

            //imshow("Eroded video", erodedFrame);

            // Search for the largest white spot with region growing
            Point pbf, pja;
            Rect roi = Rect(0,0,0,0);
            int roiSize = 0;
            int nrRect = 0;

            for (int x = 0; x < erodedFrame.rows; x++) {
                for (int y = 0; y < erodedFrame.cols; y++) {
                    if (erodedFrame.at<uchar>(x, y) > 128) {
                        int res = regionGrowing(erodedFrame, Point(y, x), pbf, pja);
                        if (res > 500) {
                            if (!nrRect || res > roiSize) {
                                roi.x = pbf.x;
                                roi.y = pbf.y;
                                roi.width = pja.x - pbf.x + 1;
                                roi.height = pja.y - pbf.y + 1;
                                roiSize = res;
                            }
                            ++nrRect;
                        }
                    }
                }
            }

            // draw yellow rectangle which represents the largest white spot
            if (nrRect > 0) {
                    rectangle(frame, Point(roi.x, roi.y), Point(roi.x + roi.width, roi.y + roi.height), Scalar(0, 255, 255, 0), 2);
            }

            // Display the frame with detected regions
            imshow("Motion detection", frame);

            if (waitKey(16) == 32) {
                cap.release();
                destroyAllWindows();
                break;
            }
        }

        break;
    }
}

int main() {
	lab10();
    return 0;
}
