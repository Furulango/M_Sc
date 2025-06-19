#include <qrt8.h>
#include <digio.h>
#include <usb.h>
#include <stdio.h>
#include <servo2d.h>
#include <interrupt.h>
#include <uart.h>
#include <lsm6ds3.h>
#include <stdlib.h>

#include <xQuP01v0.h>

#define VdKp 20
#define VdKi 10
#define ViKp 20
#define ViKi 10

// PD controller coefficients
#define Kp  6.5f
#define Kd  94.8f

// Maximum travel speed
#define VaMax 1200.f

int Vdr,Vir;
int D;
long posD,posI;
int ax,ay,az,gx,gy,gz;
int smp,motD,motI;

// Variables para monitoreo de lazos
int errorD, errorI;      // Errores de velocidad
int errorPos;            // Error de posición
int uD, uI;              // Salidas del controlador (PWM a motores)
float dPos;              // Salida del controlador de posición
int Va;                  // Velocidad adaptativa

int sFlg = 0;
int monitorMode = 0;     // 0=telemetría completa, 1=solo lazos, 2=solo posición, 3=solo velocidad

void speedCtrl(void);
void posCtrl(void);
void readSensors(void);
void processCommands(void);

void init(void){
    // IR led half power
    qrt8_led_on(128); 
    // Enable PWMs
    srv2_enable(srv2AXIS1|srv2AXIS2);

    int_init();
    int_connect(INT_TIMER0,speedCtrl);
    int_connect(INT_TIMER1,posCtrl);
    int_connect(INT_TIMER2,readSensors);
    int_global_enable();
    
    lsm6_init();
    // Config at +-4G
    lsm6_set_cs(LSM6_CS);
    lsm6_reg_write(LSM6_REG_CTRL1_XL, LSM6_ODR1K66 | LSM6_4G | LSM6_AF100);
    lsm6_clr_cs(LSM6_CS);
    
    // Config at +-1000dps
    lsm6_set_cs(LSM6_CS);
    lsm6_reg_write(LSM6_REG_CTRL2_G, LSM6_ODR1K66 | LSM6_1000dps);
    lsm6_clr_cs(LSM6_CS);
    
    timer_period(0,0.001f);
    timer_period(1,0.010f);
    timer_period(2,0.006f);
    
    uart_setup(UART_B115200);
    uart_flush();
    
    // Mensaje de inicialización
    uart_println("=== ROBOT SEGUIDOR DE LINEA ===");
    uart_println("Comandos:");
    uart_println("X = Start/Resume");
    uart_println("R = Stop");
    uart_println("M0 = Telemetria completa");
    uart_println("M1 = Solo lazos");
    uart_println("M2 = Solo posicion");
    uart_println("M3 = Solo velocidad");
    uart_println("===============================");
}

int getLine(void){
    float d,m;
    float d0,d1,d2,d3,d4,d5,d6,d7;

    // Read sensors
    qrt8_read();

    d7 = qrt8_sensor(0)>>4;
    d6 = qrt8_sensor(1)>>4;
    d5 = qrt8_sensor(2)>>4;
    d4 = qrt8_sensor(3)>>4;
    d3 = qrt8_sensor(4)>>4;
    d2 = qrt8_sensor(5)>>4;
    d1 = qrt8_sensor(6)>>4;
    d0 = qrt8_sensor(7)>>4;
    
    // Exponential line ponderation
    d = -512*d0-256*d1-128*d2-64*d3+64*d4+128*d5+256*d6+512*d7;
    m = d0+d1+d2+d3+d4+d5+d6+d7;
    return d/m;
}

void speedCtrl(void){
    static int eDi=0,eIi=0;  // Términos integrales
    
    // Errores de velocidad
    errorD = Vdr - uD;  // Comparamos referencia con salida anterior
    errorI = Vir - uI;
    
    // Acumular términos integrales (con ganancia reducida)
    eDi += errorD;
    eIi += errorI;
    
    // Controlador P + I muy suave
    uD = 2*errorD + eDi/10;  // Kp=2, Ki=0.1
    uI = 2*errorI + eIi/10;
    
    // Anti-windup simple
    if(uD > 2047) { 
        uD = 2047; 
        eDi -= errorD; 
    }
    else if(uD < -2047) { 
        uD = -2047; 
        eDi -= errorD; 
    }
    
    if(uI > 2047) { 
        uI = 2047; 
        eIi -= errorI; 
    }
    else if(uI < -2047) { 
        uI = -2047; 
        eIi -= errorI; 
    }
    
    srv2_pwm(uD,uI);
}

void posCtrl(void){
    float d,e,da;
    static float ep=0;
    
    // Read line position
    D = getLine();
    e = -D;
    da = D;
    if(da < 0) da = -da;  // Valor absoluto
    
    // Error de posición para monitoreo
    errorPos = D;
    
    // PD controller
    d = (Kp+Kd)*e-Kd*ep;
    ep = e;
    dPos = d;  // Guardar para monitoreo
    
    // Adjust travel speed
    Va = VaMax-0.0028f*da*da;
    
    // Scale correction intensity
    d *= Va/VaMax;
    
    Vdr = Va - (int)d;
    Vir = Va + (int)d;
}

void readSensors(void){
    static int k=0;
    static long axp=0,ayp=0,azp=0;
    static int axt=0,ayt=0,azt=0;
    static long gxp=0,gyp=0,gzp=0;
    static int gxt=0,gyt=0,gzt=0;
    static int mdt=0,mit=0;
    
    k++;
    if(k==8){
        k = 0;
        // Current sample
        smp = timer_get_sample(2)>>3;
        
        // Read position (menos crítico para telemetría)
        srv2_pos(posD,posI);
        
        // Promediar valores acumulados
        ax = axp>>3; axp=0;
        ay = ayp>>3; ayp=0;
        az = azp>>3; azp=0;
        
        gx = gxp>>3; gxp=0;
        gy = gyp>>3; gyp=0;
        gz = gzp>>3; gzp=0;
        
        motD = mdt>>3; mdt=0;
        motI = mit>>3; mit=0;
        
        sFlg = 1;
    }
    
    // Sample the accelerometer
    lsm6_read();
    lsm6_accel(axt,ayt,azt);
    lsm6_gyro(gxt,gyt,gzt);
    axp += axt; ayp += ayt; azp += azt;
    gxp += gxt; gyp += gyt; gzp += gzt;
    
    // Motor power
    mdt += Vdr;
    mit += Vir;
}

void processCommands(void){
    char ch;
    static char cmd[3];
    static int cmdIndex = 0;
    
    while(uart_kb_hit()){
        ch = uart_pop_data();
        
        if(ch == 'X'){
            srv2_enable(srv2AXIS1|srv2AXIS2);
            timer_start(0);
            timer_start(1);
            timer_start(2);
            uart_println(">>> INICIADO");
        }
        else if(ch == 'R'){
            timer_stop(0);
            timer_stop(1);
            timer_stop(2);
            srv2_pwm(0,0);
            srv2_enable(0);
            uart_println(">>> DETENIDO");
        }
        else if(ch == 'M'){
            cmdIndex = 0;
            cmd[cmdIndex++] = ch;
        }
        else if(cmdIndex > 0 && cmdIndex < 2){
            cmd[cmdIndex++] = ch;
            if(cmdIndex == 2){
                cmd[2] = '\0';
                if(cmd[1] >= '0' && cmd[1] <= '3'){
                    monitorMode = cmd[1] - '0';
                    uart_print(">>> Modo cambiado a: ");
                    uart_println(cmd);
                }
                cmdIndex = 0;
            }
        }
    }
}

void sendTelemetry(void){
    char num[16];
    
    switch(monitorMode){
        case 0: // Telemetría completa
            uart_print("FULL:");
            itoa10(smp, num); uart_print(num); uart_print(",");
            itoa10(D, num); uart_print(num); uart_print(",");
            itoa10(errorPos, num); uart_print(num); uart_print(",");
            itoa10((int)dPos, num); uart_print(num); uart_print(",");
            itoa10(Va, num); uart_print(num); uart_print(",");
            itoa10(Vdr, num); uart_print(num); uart_print(",");
            itoa10(uD, num); uart_print(num); uart_print(",");    // PWM motor derecho
            itoa10(errorD, num); uart_print(num); uart_print(",");
            itoa10(uD, num); uart_print(num); uart_print(",");
            itoa10(Vir, num); uart_print(num); uart_print(",");
            itoa10(uI, num); uart_print(num); uart_print(",");    // PWM motor izquierdo
            itoa10(errorI, num); uart_print(num); uart_print(",");
            itoa10(uI, num); uart_print(num);
            uart_println("");
            break;
            
        case 1: // Solo lazos principales
            uart_print("POS_REF:");
            itoa10(0, num); uart_println(num);
            uart_print("POS_ACT:");
            itoa10(D, num); uart_println(num);
            uart_print("POS_ERR:");
            itoa10(errorPos, num); uart_println(num);
            uart_print("POS_OUT:");
            itoa10((int)dPos, num); uart_println(num);
            uart_print("VEL_D_REF:");
            itoa10(Vdr, num); uart_println(num);
            uart_print("VEL_D_ACT:");
            itoa10(uD, num); uart_println(num);      // PWM enviado al motor
            uart_print("VEL_D_ERR:");
            itoa10(errorD, num); uart_println(num);
            uart_print("VEL_I_REF:");
            itoa10(Vir, num); uart_println(num);
            uart_print("VEL_I_ACT:");
            itoa10(uI, num); uart_println(num);      // PWM enviado al motor
            uart_print("VEL_I_ERR:");
            itoa10(errorI, num); uart_println(num);
            uart_println("---");
            break;
            
        case 2: // Solo posición
            uart_print("POS:");
            itoa10(D, num); uart_print(num);
            uart_print(",ERR:");
            itoa10(errorPos, num); uart_print(num);
            uart_print(",CORR:");
            itoa10((int)dPos, num); uart_print(num);
            uart_print(",VEL:");
            itoa10(Va, num); uart_println(num);
            break;
            
        case 3: // Solo velocidades (PWM)
            uart_print("VEL_D:");
            itoa10(Vdr, num); uart_print(num);
            uart_print("/");
            itoa10(uD, num); uart_print(num);        // PWM motor derecho
            uart_print("/");
            itoa10(errorD, num); uart_print(num);
            uart_print(",VEL_I:");
            itoa10(Vir, num); uart_print(num);
            uart_print("/");
            itoa10(uI, num); uart_print(num);        // PWM motor izquierdo
            uart_print("/");
            itoa10(errorI, num); uart_println(num);
            break;
    }
}

void main(void){
    init();
    
    for(;;){
        _Led = D;
        
        processCommands();
        
        if((_Btn&2)){
            srv2_enable(srv2AXIS1|srv2AXIS2);
            timer_start(0);
            timer_start(1);
            timer_start(2);
        }
        
        if(sFlg){
            sFlg = 0;
            sendTelemetry();
        }
    }
}