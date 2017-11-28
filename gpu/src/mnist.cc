// read MNIST data into an Armadillo mat
// free to use this code for any purpose
// original author : Eric Yuan
// my blog: http://eric-yuan.me/
// part of this code is stolen from http://compvisionlab.wordpress.com/
// Modified by Ilan Dubois

#include "mnist.hh"

int Mnist::ReverseInt (int i)
{
  unsigned char ch1, ch2, ch3, ch4;
  ch1 = i & 255;
  ch2 = (i >> 8) & 255;
  ch3 = (i >> 16) & 255;
  ch4 = (i >> 24) & 255;
  return((int) ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

mat Mnist::read_Mnist(std::string filename, cublasHandle_t handle)
{
  auto vec = std::vector<std::vector<double>>();
  std::ifstream file (filename, std::ios::binary);
  if (file.is_open())
  {
    int magic_number = 0;
    int number_of_images = 0;
    int n_rows = 0;
    int n_cols = 0;
    file.read((char*) &magic_number, sizeof (magic_number));
    magic_number = ReverseInt(magic_number);
    file.read((char*) &number_of_images,sizeof (number_of_images));
    number_of_images = ReverseInt(number_of_images);
    file.read((char*) &n_rows, sizeof (n_rows));
    n_rows = ReverseInt(n_rows);
    file.read((char*) &n_cols, sizeof (n_cols));
    n_cols = ReverseInt(n_cols);
    for (int i = 0; i < number_of_images; ++i)
    {
      std::vector<double> tp;
      for (int r = 0; r < n_rows; ++r)
      {
        for (int c = 0; c < n_cols; ++c)
        {
          unsigned char temp = 0;
          file.read((char*) &temp, sizeof (temp));
          tp.push_back((double)temp / 255.);
        }
      }
      vec.push_back(tp);
    }
  }

  float* data = new float[MNIST_IMG_SIZE * vec.size()];
  for (size_t i = 0; i < vec.size(); ++i) {
    for (size_t j = 0; j < MNIST_IMG_SIZE; ++j)
      data[i * MNIST_IMG_SIZE + j] = (float)vec[i][j];
  }
  auto ans = mat(handle, vec.size(), MNIST_IMG_SIZE, data, true);
  delete data;
  return ans;
}

mat Mnist::read_Mnist_Label(std::string filename, cublasHandle_t handle, size_t nb)
{
  auto vec = std::vector<double>(nb);
  std::ifstream file (filename, std::ios::binary);
  if (file.is_open())
  {
    int magic_number = 0;
    int number_of_images = 0;
    int n_rows = 0;
    int n_cols = 0;
    file.read((char*) &magic_number, sizeof (magic_number));
    magic_number = ReverseInt(magic_number);
    file.read((char*) &number_of_images,sizeof (number_of_images));
    number_of_images = ReverseInt(number_of_images);
    for (int i = 0; i < number_of_images; ++i)
    {
      unsigned char temp = 0;
      file.read((char*) &temp, sizeof (temp));
      vec[i]= (double)temp;
    }
  }
  float* data = new float[NB_CLASS * vec.size()]();
  for (size_t i = 0; i < vec.size(); ++i)
    data[NB_CLASS *  i + (int)vec[i]] = 1.;
  auto ans =  mat(handle, vec.size(), NB_CLASS, data, true);
  delete data;
  return ans;
}
