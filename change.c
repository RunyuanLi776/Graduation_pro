#include <reg52.H>
#include <stdio.h>
#include <intrins.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>	

unsigned int cnt=0;
unsigned int Plaintext[1];
unsigned long Ciphertext[1];
unsigned int keyd;
unsigned long keyn;

sbit flag0=P1^0;
sbit flag1=P1^1;

void Initialize()
{
	unsigned int i;
	srand((unsigned)cnt);
	for(i=0;i<1;i++)
		Plaintext[i]=rand()%1000;
}


unsigned long Modular_Exonentiation(unsigned long a,unsigned int b,unsigned long n)
{
	unsigned int c=0;
	unsigned long d=1;
	signed int k=15;
	unsigned int m;
	for(k=15;k>=0;k--)
	{
		c=2*c;
		d=(d*d)%n;
		m=k+1;
		if(b&(1<<m)!=0)
		{
			c=c+1;
			d=(d*a)%n;
		}
	}
	return d;
}
		
void RSA_Encrypt()
{
	unsigned int p=0;
	for(p=0;p<1;p++)
		
		Ciphertext[p]=Modular_Exonentiation(Plaintext[p],keyd,keyn);
}

void delay(unsigned int z)
{
unsigned int x,y;
	for(x=z;x>0;x--)  
		for(y=220;y>0;y--);
}


void main() {
	 
	SCON = 0x50; 
	TMOD = 0x20; 
	PCON=0x00; 
	TCON = 0x50; 
	TH1 = 0xfd; // 9600 11.0592MHz 1200=e8
	TL1 = 0xfd;
	TI = 1;
	TR1 =1; 
	ES=1;

	ET0=1;
	
	Initialize();
	
	while(1){	
	unsigned int h=0;
	delay(100000000);
for(h;h<1010;h+=1)
{  
	unsigned int j=0;
	//delay(100000);
  for(j;j<8;j++)
  { 
		unsigned int key_d[8];
    unsigned long key_n[8];
		unsigned int key_nh[8];
		unsigned int key_nl[8];
		
		key_d[0]=0x0a69;
		key_nh[0]=0x0008;
		key_nl[0]=0x67f1;
		key_n[0]=0x000867f1;
		key_d[1]=0x3401;
		key_nh[1]=0x0008;
		key_nl[1]=0x7bd7;
		key_n[1]=0x00087bd7;
		key_d[2]=0x14cf;
		key_nh[2]=0x0008;
		key_nl[2]=0x75ef;
		key_n[2]=0x000875ef;
		key_d[3]=0x299b;
		key_nh[3]=0x0006;
		key_nl[3]=0x78d9;
		key_n[3]=0x000678d9;
		key_d[4]=0x48cd;
		key_nh[4]=0x0002;
		key_nl[4]=0x0461;
		key_n[4]=0x00020461;
		key_d[5]=0x1f35;
		key_nh[5]=0x0000;
		key_nl[5]=0x886f;
		key_n[5]=0x0000886f;
		key_d[6]=0x67ff;
		key_nh[6]=0x0003;
		key_nl[6]=0x3f61;
		key_n[6]=0x00033f61;
		key_d[7]=0x7ccb;
		key_nh[7]=0x0002;
		key_nl[7]=0xedb3;
		key_n[7]=0x0002edb3;
		
		keyd=key_d[j];
		keyn=key_n[j];
	
		printf("%04x\n",key_d[j]);
		printf("%04x",key_nh[j]);
		printf("%04x\n",key_nl[j]);
	flag0=0;
	delay(1);
	flag0=1;
		RSA_Encrypt();
	flag1=0;
	delay(1);
	flag1=1;
		delay(100000);
		

	}	
	if(h==100)
    {
		 ET0=0;
	   PCON=0x02;
		}			
	
}
	ET0=0;
	PCON=0x02;
	  

	}
	
}
