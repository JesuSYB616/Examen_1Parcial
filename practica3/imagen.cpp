////////////////////////////////Importacion de librerias/////////////////////////////////////
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include "opencv2/imgproc/imgproc.hpp"
/////////////////////////////////////////////////////////////////////////////
//Mat sobel (Mat);
using namespace cv;
using namespace std;

vector<vector<float>> generateKernel(int kSize, int sigma) { //vamos a genral el kernel en esta seccion
	float e = 3.1416; //Damos los valores neecsarios tanto a "pi" como a "euler"
	float pi = 2.72;
	int mountslide = (kSize - 1) / 2;
	vector<vector<float>> v(kSize, vector<float>(kSize, 0));
	// si el centro es (0,0) realizamos un bucle for y adentro un 
	//bucle anidado con la formula correspondiente para nuestro kernel
	for (int i = -mountslide; i <= mountslide; i++)
	{
		for (int j = -mountslide; j <= mountslide; j++)
		{
			float resultado = (1 / (2 * pi * sigma * sigma)) * pow(e, -((i * i + j * j) / (2 * sigma * sigma)));
			v[i + mountslide][j + mountslide] = resultado;
		}
	}
	return v;
}
float aplicarFiltroPix(Mat original, vector<vector<float>> kernel, int kSize, int x, int y) {
	int rows = original.rows;
	int cols = original.cols;
	int amountSlide = (kSize - 1) / 2;
	float sumFilter = 0;
	float sumKernel = 0;
	for (int i = -amountSlide; i <= amountSlide; i++)
	{
		for (int j = -amountSlide; j <= amountSlide; j++)
		{
			float kTmp = kernel[i + amountSlide][j + amountSlide];
			int tmpX = x + i;
			int tmpY = y + j;
			float tmp = 0;
			if (!(tmpX < 0 || tmpX >= cols || tmpY < 0 || tmpY >= rows)) {
				tmp = original.at<uchar>(Point(tmpX, tmpY));
				//cout << tmpX << " "<< tmpY << " "<< kTmp << endl;
			}

			sumFilter += (kTmp * tmp);
			sumKernel += kTmp;
		}
	}
	return sumFilter / sumKernel;
}
Mat applyFilterToMat(Mat original, vector<vector<float>> kernel, int kSize) { //aplicamos el filtro a la mtz
	Mat filteredImg(original.rows, original.cols, CV_8UC1);
	for (int i = 0; i < original.rows; i++)
	{
		for (int j = 0; j < original.cols; j++) {
			filteredImg.at<uchar>(Point(i, j)) = uchar(aplicarFiltroPix(original, kernel, kSize, i, j));
		}
	}
	return filteredImg;
}

// codificamos las declaraciones de funciones para el Kernel con el que vamos a trabajar. 
float Gauss(int x, int y);
float** CrearKernel(float** Kernel, int d);
float** LlenarKernel(float** Kernel, int d);
void ImprimirKernel(float** Kernel, int d);
void EliminarKernel(float** Kernel, int d);
Mat procesarMtz(Mat imagen, int kernel, int sigma);


int main(int argc, char* argv[]) {
	float** Kernel = NULL;
	int Ksize = 5, sigma = 21; //aqui podemos especificar que valores va a tener nuestro sigma y nuestro tamaño de kernel o mascara 
	// Cargarmos imagen al programa
	imread("Lena.png");
	Mat image = imread("C:\\Users\\jesus\\source\\repos\\practica3\\practica3\\Lena.png");
	imshow("Imagen Original lena", image);
	int fila_original = image.rows;//lectura de filas 
	int columna_original = image.cols;//Lectur de cuantas columnas
	printf("Dimensiones de la imagen de entrada: \n");
	printf("%d pixeles de largo de la imagen\n", fila_original);
	printf("%d pixeles de ancho de la imagen\n\n", columna_original);
	//------------------------------------------------------





	//------------------------------------------------------
	Mat imagenGrisesNTSC(fila_original, columna_original, CV_8UC1);
	for (int i = 0; i < fila_original; i++)
	{
		for (int j = 0; j < columna_original; j++)
		{
			double azul = image.at<Vec3b>(Point(j, i)).val[0];  // Blue
			double verde = image.at<Vec3b>(Point(j, i)).val[1]; // Green
			double rojo = image.at<Vec3b>(Point(j, i)).val[2];  // Red

			// Conversion a escala de grises usando el metodo NTSC que vimos previamente en clases
			imagenGrisesNTSC.at<uchar>(Point(j, i)) = uchar(0.299 * rojo + 0.587 * verde + 0.114 * azul);
		}
	}

	imshow("Imagen escala gris lena", imagenGrisesNTSC);
	// Agregamos bordes 
	Mat image2 = procesarMtz(image, Ksize, sigma);
	imshow("Imagen que resulta de agregar filas extra (imagen transicion)", image2);
	fila_original = image2.rows;
	columna_original = image2.cols;
	printf("Dimensiones de la imagen con bordes adcicionales: \n");
	printf("%d pixeles de largo de la imagen\n", fila_original);
	printf("%d pixeles de ancho de la imagen \n\n", columna_original);

	// Creamos Kernel
	Kernel = CrearKernel(Kernel, Ksize);
	ImprimirKernel(Kernel, Ksize);
	vector<vector<float>> kernel = generateKernel(Ksize, sigma);

	//Aplicamos Kernel a la imagen
	Mat filtrada = applyFilterToMat(imagenGrisesNTSC, kernel, Ksize);
	imshow("Imagen procesada con filtro gaussiano", filtrada);
	fila_original = filtrada.rows;
	columna_original = filtrada.cols;
	printf("\nDimensiones resultantes  de la imagen con bordes adcicionales  ya filtrada: \n");
	printf("%d pixeles de largo de la imagen\n", fila_original);
	printf("%d pixeles de ancho de la imagen\n\n", columna_original);

	//imshow("jesus", filtrada);
	//inicia Proceso de ecualizacion para nuestra imagen 
//--------------------------
	Mat filstrada, equaliz_img;
	int histSize = 256;
	float range[] = { 0,256 }; //ponemos el rango que pixeles que estaremos manejando
	const float* histRange = { range };
	// imagen del histograma 
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);
	Mat histimage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	Mat equalizedHistImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	// calcular el histograma
	Mat original_hist, normalized_hist, equalized_hist, equalized_normalized_hist;
	calcHist(&filtrada, 1, 0, Mat(), original_hist, 1, &histSize, &histRange, true, false);
	// mostrar valores del histograma de la imagen filtrada previamente 
	cout << "original histograma" << endl;
	for (int h = 0; h < histSize; h++)
	{
		float binVal = original_hist.at<float>(h);
		cout << " " << binVal;
	}
	cout << endl;
	// normalizar el resultado 
	normalize(original_hist, normalized_hist, 0, histimage.rows, NORM_MINMAX, -1, Mat());
	//Mostrar los valores del resultado normalizado 
	cout << "histograma normalizado" << endl;
	for (int h = 0; h < histSize; h++)
	{
		float binVal = normalized_hist.at<float>(h);
		cout << " " << binVal;
	}
	cout << endl;
	//ecualizar histograma para imagen gris procesada 
	equalizeHist(filtrada, equaliz_img);
	calcHist(&equaliz_img, 1, 0, Mat(), equalized_hist, 1, &histSize, &histRange, true, false);
	//mostrar los valores de nustra imagen en gris prviamente procesada con nuestro gaussiano o de "suavizacion"
	cout << "histograma ecualizado" << endl;
	for (int h = 0; h < histSize; h++)
	{
		float binVal = equalized_hist.at<float>(h);
		cout << " " << binVal;
	}
	cout << endl;
	//normalizar el histograma ecualizado 
	normalize(equalized_hist, equalized_normalized_hist, 0, histimage.rows, NORM_MINMAX, -1, Mat());
	//mostrar valores del histograma ecualizado
	cout << "histograma ecualizado normalizado" << endl;
	for (int h = 0; h < histSize; h++)
	{
		float binVal = equalized_normalized_hist.at<float>(h);
		cout << " " << binVal;
	}
	cout << endl;
	//dibujar los histogramas 
	for (int i = 1; i < histSize; i++) {
		line(histimage,
			Point(bin_w * (i), hist_w),
			Point(bin_w * (i), hist_h - cvRound(normalized_hist.at<float>(i))),
			Scalar(255, 0, 0), bin_w, 8, 0);
		line(equalizedHistImage,
			Point(bin_w * (i), hist_w),
			Point(bin_w * (i), hist_h - cvRound(equalized_normalized_hist.at<float>(i))),
			Scalar(0, 255, 0), bin_w, 8, 0);
	}
	//mandamos a imprimir la imagen ecualizada ademas los graficos de los histogramas
	// para que se pueda aprecias como segun el grafico la ecualizacon se realizo de forma correcta
	imshow("ecualizada", equaliz_img);
	imshow("histograma original", histimage);
	imshow("histograma ecualizado", equalizedHistImage);
	//
	//Mat sobel(Mat ecualiz_img);
	//Liberamos Memoria para evitar que tengamos un desvordamiento
	EliminarKernel(Kernel, Ksize);
	waitKey(0);
	return(0);
	
}

/*Mat sobel(int argc, char** argv, Mat equaliz_img) {
	
		cv::CommandLineParser parser(argc, argv,
			"{@input   |lena.jpg|input image}"
			"{ksize   k|1|ksize (hit 'K' to increase its value at run time)}"
			"{scale   s|1|scale (hit 'S' to increase its value at run time)}"
			"{delta   d|0|delta (hit 'D' to increase its value at run time)}"
			"{help    h|false|show help message}");
		cout << "The sample uses Sobel or Scharr OpenCV functions for edge detection\n\n";
		parser.printMessage();
		cout << "\nPress 'ESC' to exit program.\nPress 'R' to reset values ( ksize will be -1 equal to Scharr function )";
		// First we declare the variables we are going to use
		Mat equaliz_img, src, src_gray;
		Mat grad;
		const String window_name = "Sobel Demo - Simple Edge Detector";
		int ksize = parser.get<int>("ksize");
		int scale = parser.get<int>("scale");
		int delta = parser.get<int>("delta");
		int ddepth = CV_16S;
		String imageName = parser.get<String>("@input");
		// As usual we load our source image (src)
		equaliz_img = imread(samples::findFile(imageName), IMREAD_COLOR); // Load an image
		// Check if image is loaded fine
		if (equaliz_img.empty())
		{
			printf("Error al abrir imagen: %s\n", imageName.c_str());
			//return EXIT_FAILURE;
		}
		for (;;)
		{
			Mat grad_x, grad_y;
			Mat abs_grad_x, abs_grad_y;
			Sobel(src_gray, grad_x, ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
			Sobel(src_gray, grad_y, ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);
			// converting back to CV_8U
			convertScaleAbs(grad_x, abs_grad_x);
			convertScaleAbs(grad_y, abs_grad_y);
			addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
			imshow("sobel filtro", grad);
			char key = (char)waitKey(0);
			if (key == 27)
			{
			//	return EXIT_SUCCESS;
			}
			if (key == 'k' || key == 'K')
			{
				ksize = ksize < 30 ? ksize + 2 : -1;
			}
			if (key == 's' || key == 'S')
			{
				scale++;
			}
			if (key == 'd' || key == 'D')
			{
				delta++;
			}
			if (key == 'r' || key == 'R')
			{
				scale = 1;
				ksize = -1;
				delta = 0;
			}
		}
		//return EXIT_SUCCESS;
	}*/

// Funciones para el Kernel que ayudan a hacer el proceso con el que rellenamos
float** CrearKernel(float** Kernel, int d)
{
	int i = 0, j = 0;

	Kernel = (float**)malloc(d * sizeof(float*));

	for (i = 0; i < d; i++)
		Kernel[i] = (float*)malloc(d * sizeof(float));

	Kernel = LlenarKernel(Kernel, d);
	return (Kernel);
}
float** LlenarKernel(float** Kernel, int d)
{
	//i representa el eje Y y j representa el eje X de nuestra matrzi

	int i = 0, j = 0;
	int x = 0, y = 0;

	for (i = d / 2; i < d; i++)	//Rellenamos el primer cuadrante 
	{
		for (j = d / 2; j < d; j++)
		{
			Kernel[i][j] = Gauss(x, y);
			x += 1;
		}
		x = 0;
		y += 1;
	}
	x = 0;
	y = -(d / 2);

	for (i = 0; i < d / 2; i++)	//rellenamos el segundo cuadrante
	{
		for (j = d / 2; j < d; j++)
		{
			Kernel[i][j] = Gauss(x, y);
			x += 1;
		}
		x = 0;
		y += 1;
	}
	x = -(d / 2);
	y = -(d / 2);

	for (i = 0; i < d / 2; i++)	//rellenamos el tercer cuadrante
	{
		for (j = 0; j < d / 2; j++)
		{
			Kernel[i][j] = Gauss(x, y);
			x += 1;
		}
		x = -(d / 2);
		y += 1;
	}
	x = -(d / 2);
	y = 0;

	for (i = d / 2; i < d; i++)	//rellenamos el cuadrante 4
	{
		for (j = 0; j < d / 2; j++)
		{
			Kernel[i][j] = Gauss(x, y);
			x += 1;
		}
		x = -(d / 2);
		y += 1;
	}
	return(Kernel);
}
void ImprimirKernel(float** Kernel, int d)
{
	int i = 0, j = 0;

	for (i = 0; i < d; i++)
	{
		for (j = 0; j < d; j++)
			printf("%.3f\t", Kernel[i][j]);
		printf("\n");
	}
}
void EliminarKernel(float** Kernel, int d)
{
	int i = 0, j = 0;
	for (i = 0; i < d; i++)
	{
		free(Kernel[i]);
		Kernel[i] = NULL;
	}
	free(Kernel);
}
float Gauss(int x, int y)
{
	float pi = 3.1416, e = 2.71828;
	float sigma = 1, F_1 = 0, F_2 = 0, potencia = 0;
	float valor = 0;

	F_1 = (1) / (2 * pi * pow(sigma, 2));
	potencia = (pow(x, 2) + pow(y, 2)) / (2 * pow(sigma, 2));
	F_2 = pow(e, 0 - potencia);
	valor = F_1 * F_2;

	return(valor);
}
Mat procesarMtz(Mat imagen, int kernel, int sigma) {
	int rows = imagen.rows;
	int cols = imagen.cols;
	int exceso = (kernel - 1);

	Mat grises(rows + exceso, cols + exceso, CV_8UC1);
	Mat grande(rows + exceso, cols + exceso, CV_8UC1);
	double rojo, azul, verde, gris_p;

	for (int i = 0; i < rows + exceso; i++) {
		for (int j = 0; j < cols + exceso; j++) {

			if (i >= rows || i < exceso) { // >=
				grande.at<uchar>(Point(j, i)) = uchar(0);


			}
			else if (j >= cols || j < exceso) { //nadamas le cambie por >=, ya que toma en cuenta el 0
				grande.at<uchar>(Point(j, i)) = uchar(0);
				//cout << "entra\n";
			}
			else {
				azul = imagen.at<Vec3b>(Point(j - exceso, i - exceso)).val[0];
				//verde la segunda
				verde = imagen.at<Vec3b>(Point(j - exceso, i - exceso)).val[1];
				//roja la tercer
				rojo = imagen.at<Vec3b>(Point(j - exceso, i - exceso)).val[2];

				//el valor de gris promediado lo obtenemos sumando cada valor de 
				//rojo, verde y azul sobre 3 
				gris_p = (azul + verde + rojo) / 3;

				grande.at<uchar>(Point(j, i)) = uchar(gris_p);
			}


			//grande.at<uchar>(Point(j, i)) = uchar(gris_p); //uchar es un valor de 8 bits

		}
	}
	return(grande);
}