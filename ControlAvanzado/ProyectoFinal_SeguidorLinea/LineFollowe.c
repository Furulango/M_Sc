#include <qrt8.h>
#include <digio.h>
#include <usb.h>
#include <stdio.h>
#include <servo2d.h>
#include <interrupt.h>

#include <xQuP01v0.h>

#define VdKp 20
#define VdKi 10
#define ViKp 20
#define ViKi 10

#define Kp 	0.01000f
#define Ki 	0.000035f
#define Kd 	0.0000f

//#define Kp 	1.75f
//#define Ki 	0.001f

//#define Kp 	1.75f
//#define Ki 	0.001f
//#define Kd 	0.02f

//#define a0 (Kp+Ki)
//#define a1 (-Kp)
//#define a2 0

#define a0 (Kp+Ki+Kd)
#define a1 (-Kp-2*Kd)
#define a2 Kd

//#define DTH 2500
int velD,velI;
int Vdr,Vir;
int D;
//int Va = 1000;
int Va = 7;

void speedCtrl(void);
void posCtrl(void);

void init(void){

	// IR led half power
	qrt8_led_on(128); 
	// Enable PWMs
	srv2_enable(srv2AXIS1|srv2AXIS2);

	int_init();
	int_connect(INT_TIMER0,speedCtrl);
	int_connect(INT_TIMER1,posCtrl);
	int_global_enable();
	
//	timer_period(0,0.01f);
//	timer_period(1,0.05f);
	timer_period(0,0.02f);
	timer_period(1,0.021f);
}

int getLine(void){
	float d,m;
	float d0,d1,d2,d3,d4,d5,d6,d7;

	qrt8_read();

	d7 = qrt8_sensor(0)>>4;
	d6 = qrt8_sensor(1)>>4;
	d5 = qrt8_sensor(2)>>4;
	d4 = qrt8_sensor(3)>>4;
	d3 = qrt8_sensor(4)>>4;
	d2 = qrt8_sensor(5)>>4;
	d1 = qrt8_sensor(6)>>4;
	d0 = qrt8_sensor(7)>>4;
	
//	d = -64*d0-32*d1-16*d2-8*d3+8*d4+16*d5+32*d6+64*d7;
	d = -320*d0-280*d1-240*d2-200*d3+200*d4+240*d5+280*d6+320*d7;
	m = d0+d1+d2+d3+d4+d5+d6+d7;
	return d/m;
}

void speedCtrl(void){
	int vD,vI;
	int uD,uI;
	int eD,eI;
	static int eDp=0,eIp=0;
	static int uDp=0,uIp=0;
	static int d1=0,d2=0,d3=0,d4=0;
	static int i1=0,i2=0,i3=0,i4=0;
			
	srv2_vel(vD,vI);
	
	d4=d3; d3=d2; d2=d1; d1=vD;
	velD = d4+d3+d2+d1;
	
	i4=i3; i3=i2; i2=i1; i1=vI;
	velI = i4+i3+i2+i1;
	
	eD = Vdr - velD;
	eI = Vir - velI;
	
	uD = (VdKp+VdKi)*eD-VdKp*eDp+uDp;
	uI = (ViKp+ViKi)*eI-ViKp*eIp+uIp;
	
	eDp = eD;
	eIp = eI;
	uDp = uD;
	uIp = uI;

//	uD = Vdr;
//	uI = Vir;
	
	if(uD<-2047)
		uD=-2047;
	else if(uD>2047)
		uD=2047;
	if(uI<-2047)
		uI=-2047;
	else if(uI>2047)
		uI=2047;
	
	srv2_pwm(uD,uI);
}

void posCtrl(void){
	float d,e;
	static float dp=0,ep=0,epp=0;
	
	D = getLine();
	e = -D;
	
//	d = Kp*e;
	
	d = a0*e+a1*ep+a2*epp+dp;
	epp = ep;
	ep = e;
	dp = d;
	
	Vdr = Va - (int)d;
	Vir = Va + (int)d;
}

void main(void){
	char str[256];
	
	init();
	
//	Vdr = 10;
//	Vir = 10;
	
	for(;;){
		_Led = D;
		
		if(_Btn&2){
			timer_start(0);
			timer_start(1);
			_Led = _Btn;
		}
		
//		if(_Btn&4){
//			timer_stop(0);
//			timer_stop(1);
//			srv2_pwm(0,0);
//		}
		
//		sprintf(str,"%d %d %d\n",D,velI,velD);
////		sprintf(str,"%d %d %d\n",D,Vir,Vdr);
//		usb_print(str);
//		delay_ms(100);
	}

}