#include "mlp_coefs.h"
#include "mlp_intercepts.h"

/*
Random number generator:

Press, W.H. et al.,
Numerical Recipes in C
The Art of Scientific Computing 2nd Ed.
1992.
Chapter 7, Random Numbers.

  */

#define IA 16807
#define IM 2147483647
#define AM (1.0/IM)
#define IQ 127773
#define IR 2836
#define MASK 123459876

float ran0(long* idum)
{
  long k;
  float ans;
  *idum ^= MASK; 
  k = (*idum) / IQ; 
  *idum = IA * (*idum - k * IQ) - IR * k;
  if (*idum < 0) 
      *idum += IM;
  ans = AM * (*idum); 
  *idum ^= MASK; 
  return ans;
}

#define IMG_LEN 784
byte img[IMG_LEN];

float buff0[MLP_COEFS_ROWS];

void setup() {
  Serial.begin(9600);
  pinMode(LED_BUILTIN_TX, INPUT);
  pinMode(LED_BUILTIN_RX, INPUT);   
}


void calc_xwt(){
  long  congnum = 1;
  for (int  i = 0 ; i < MLP_COEFS_ROWS ; i++){
      float val = 0.0;
      for (int  j = 0 ; j < IMG_LEN ; j++){
        float rnd01 = ran0(&congnum);
        rnd01 -= 0.5;
        val += img[j] / 255.0 * rnd01;
      }
      buff0[i] = val;
  }
}


int calc_digit()
{
    int   idx_of_max = -1;
    float val_of_max;
    for (int   i = 0 ; i < MLP_INTERCEPTS_LEN ; i++){
        float digit_w = 0.0;
        for (int  j = 0 ; j < MLP_COEFS_ROWS ; j++){
            digit_w += MLP_COEFS[j][i] * buff0[j];
        }
        digit_w /= MLP_COEFS_FACTOR;
        digit_w += MLP_INTERCEPTS[i] / (float)MLP_INTERCEPTS_FACTOR;
        if (idx_of_max == -1 || digit_w > val_of_max){
            idx_of_max = i;
            val_of_max = digit_w;
        }
    }
    return idx_of_max;
}


int  input_idx = 0;

void loop() {
  if (Serial.available() > 0) {
    img[input_idx] = Serial.read(); 
    input_idx++;
    if (IMG_LEN == input_idx) {
      Serial.println("...0Received");
      calc_xwt();
      int Digit = calc_digit();
      Serial.print(Digit);  
      Serial.flush();
      input_idx = 0;
    }
  }
}
