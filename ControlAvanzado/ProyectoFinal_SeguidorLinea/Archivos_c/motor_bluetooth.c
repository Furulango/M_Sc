#include <qrt8.h>
#include <digio.h>
#include <usb.h>
#include <uart.h>
#include <stdio.h>
#include <servo2d.h>
#include <interrupt.h>
#include <stdlib.h>
#include <xQuP01v0.h>

// Parámetros PID 
#define VdKp 8      
#define VdKi 3      
#define VdKd 1      
#define ViKp 8      
#define ViKi 3
#define ViKd 1

// Límites para anti-windup del integrador
#define MAX_INTEGRAL 500
#define MIN_INTEGRAL -500

// Variables globales para velocidades
int velD, velI;        
int Vdr, Vir;         
int velocidad_objetivo = 8;  // Velocidad base

// Variables para compensación de motores
float compensacion_derecho = 1.0f;  
int motor_activo = 0;

// Variables para rampa suave de velocidad
int vel_actual_D = 0, vel_actual_I = 0;
int incremento_rampa = 1;  

// Variables para monitoreo
int pwm_D_actual = 0, pwm_I_actual = 0;
int contador_monitor = 0;

void speedCtrl(void);

void init(void){
    // Configurar Bluetooth
    uart_setup(UART_B115200);
    uart_println("=== ROBOT BOLA LOCA ADELANTE ===");
    uart_println("Configuracion: BOLA ADELANTE - RUEDAS EMPUJAN ATRAS");
    uart_println("BTN1: -Derecho  BTN2: ON/OFF  BTN3: +Derecho");
    uart_println("Sensores entre ruedas y bola loca");
    
    // Habilitar PWMs
    srv2_enable(srv2AXIS1|srv2AXIS2);

    // Configurar interrupciones
    int_init();
    int_connect(INT_TIMER0, speedCtrl);
    int_global_enable();
    
    // Timer cada 10ms 
    timer_period(0, 0.01f);
}

void speedCtrl(void){
    int vD, vI;
    int uD, uI;
    int eD, eI;
    static int eDp=0, eIp=0;  
    static int integralD=0, integralI=0;      
    static int uDp=0, uIp=0;
    
    // Filtro: promedio móvil de 8 muestras
    static int filtroD[8] = {0}, filtroI[8] = {0};
    static int indice = 0;
    int sumaD = 0, sumaI = 0;
    int j;
            
    // Leer velocidades actuales
    srv2_vel(vD, vI);
    
    // Filtro promedio móvil
    filtroD[indice] = vD;
    filtroI[indice] = vI;
    
    for(j = 0; j < 8; j++){
        sumaD += filtroD[j];
        sumaI += filtroI[j];
    }
    
    velD = sumaD / 8;  
    velI = sumaI / 8;  
    
    indice = (indice + 1) % 8;  
    
    // Calcular errores
    eD = Vdr - velD;
    eI = Vir - velI;
    
    // Acumular integrales con anti-windup
    integralD += eD;
    integralI += eI;
    
    // Límites del integrador
    if(integralD > MAX_INTEGRAL) integralD = MAX_INTEGRAL;
    else if(integralD < MIN_INTEGRAL) integralD = MIN_INTEGRAL;
    if(integralI > MAX_INTEGRAL) integralI = MAX_INTEGRAL;
    else if(integralI < MIN_INTEGRAL) integralI = MIN_INTEGRAL;
    
    // Controlador PID 
    uD = VdKp*eD + VdKi*integralD + VdKd*(eD - eDp);  
    uI = ViKp*eI + ViKi*integralI + ViKd*(eI - eIp);  
    
    // APLICAR COMPENSACION al motor derecho
    uD = (int)(uD * compensacion_derecho);
    
    // Filtro de salida para suavizar PWM
    uD = (uD + uDp) / 2;
    uI = (uI + uIp) / 2;
    
    // Guardar valores anteriores
    eDp = eD;
    eIp = eI;
    uDp = uD;
    uIp = uI;
    
    // Limitar salida PWM
    if(uD < -1800) uD = -1800;
    else if(uD > 1800) uD = 1800;
    if(uI < -1800) uI = -1800;
    else if(uI > 1800) uI = 1800;
    
    // Guardar PWM para monitoreo
    pwm_D_actual = uD;
    pwm_I_actual = uI;
    
    // *** CONFIGURACIÓN BOLA ADELANTE ***
    // PWM INVERTIDO: Las ruedas van hacia ATRÁS, empujando la bola hacia ADELANTE
    srv2_pwm(-uD, -uI);
}

void enviar_estado(void){
    char num[16];
    int diferencia_vel;
    
    diferencia_vel = velD - velI;
    
    uart_print("VelD:");
    itoa10(velD, num);
    uart_print(num);
    uart_print(" VelI:");
    itoa10(velI, num);
    uart_print(num);
    
    uart_print(" Dif:");
    itoa10(diferencia_vel, num);
    uart_print(num);
    
    uart_print(" Comp:");
    itoa10((int)(compensacion_derecho * 100), num);
    uart_print(num);
    uart_print("%");
    
    // Diagnóstico del comportamiento (BOLA ADELANTE)
    if(diferencia_vel >= -1 && diferencia_vel <= 1){
        uart_println(" RECTO-OK");
    }
    else if(diferencia_vel > 1){
        uart_println(" GIRA-IZQUIERDA (rueda-der mas rapida)");
    }
    else{
        uart_println(" GIRA-DERECHA (rueda-izq mas rapida)");
    }
}

void main(void){
    static int boton_anterior = 0;
    static int contador_rampa = 0;
    char num[16];
    
    init();
    
    uart_println("");
    uart_println("INSTRUCCIONES:");
    uart_println("- Pon el robot con BOLA ADELANTE");
    uart_println("- Sensores deben estar entre ruedas y bola");
    uart_println("- Las ruedas iran hacia ATRAS (empujan)");
    uart_println("- Presiona BTN2 para probar");
    
    for(;;){
        // BOTON 2: Start/Stop (toggle)
        if((_Btn & 2) && !(boton_anterior & 2)){
            if(!motor_activo){
                // Encender
                motor_activo = 1;
                vel_actual_D = 0;
                vel_actual_I = 0;
                timer_start(0);
                _Led = 0x55;
                uart_println("MOTORES ON - Bola va ADELANTE");
            }
            else{
                // Apagar
                motor_activo = 0;
                timer_stop(0);
                srv2_pwm(0, 0);
                Vdr = 0;
                Vir = 0;
                vel_actual_D = 0;
                vel_actual_I = 0;
                _Led = 0;
                uart_println("MOTORES OFF");
            }
        }
        
        // BOTON 1: Reducir motor derecho (si gira hacia izquierda)
        if((_Btn & 1) && !(boton_anterior & 1)){
            compensacion_derecho -= 0.05f;
            if(compensacion_derecho < 0.5f) compensacion_derecho = 0.5f;
            uart_print("Motor Der: ");
            itoa10((int)(compensacion_derecho * 100), num);
            uart_print(num);
            uart_println("% (REDUCIDO)");
        }
        
        // BOTON 3 (bit 4): Aumentar motor derecho (si gira hacia derecha)
        if((_Btn & 4) && !(boton_anterior & 4)){
            compensacion_derecho += 0.05f;
            if(compensacion_derecho > 1.5f) compensacion_derecho = 1.5f;
            uart_print("Motor Der: ");
            itoa10((int)(compensacion_derecho * 100), num);
            uart_print(num);
            uart_println("% (AUMENTADO)");
        }
        
        // Rampa suave cuando motores activos
        if(motor_activo){
            contador_rampa++;
            
            if(contador_rampa >= 5){  
                contador_rampa = 0;
                
                if(vel_actual_D < velocidad_objetivo){
                    vel_actual_D += incremento_rampa;
                    if(vel_actual_D > velocidad_objetivo) 
                        vel_actual_D = velocidad_objetivo;
                }
                
                if(vel_actual_I < velocidad_objetivo){
                    vel_actual_I += incremento_rampa;
                    if(vel_actual_I > velocidad_objetivo) 
                        vel_actual_I = velocidad_objetivo;
                }
                
                Vdr = vel_actual_D;
                Vir = vel_actual_I;
            }
            
            // Monitoreo cada 500ms
            contador_monitor++;
            if(contador_monitor >= 50){  
                contador_monitor = 0;
                enviar_estado();
            }
        }
        
        boton_anterior = _Btn;
        delay_ms(10);  
    }
}