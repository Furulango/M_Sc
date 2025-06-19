#include <qrt8.h>
#include <digio.h>
#include <usb.h>
#include <stdio.h>
#include <servo2d.h>
#include <interrupt.h>
#include <xQuP01v0.h>

// Parámetros del controlador PID más suaves para mayor estabilidad
#define VdKp 8      // Reducido para menos oscilación
#define VdKi 3      // Reducido para respuesta más suave
#define VdKd 1      // Añadido término derivativo para estabilidad
#define ViKp 8
#define ViKi 3
#define ViKd 1

// Límites para anti-windup del integrador
#define MAX_INTEGRAL 500
#define MIN_INTEGRAL -500

// Variables globales para velocidades
int velD, velI;        // Velocidades actuales medidas (filtradas)
int Vdr, Vir;         // Velocidades de referencia (objetivo)
int velocidad_objetivo = 15;  // Velocidad objetivo (reducida para empezar más suave)

// Variables para rampa suave de velocidad
int vel_actual_D = 0, vel_actual_I = 0;
int incremento_rampa = 1;  // Incremento gradual de velocidad

void speedCtrl(void);

void init(void){
    // Habilitar PWMs
    srv2_enable(srv2AXIS1|srv2AXIS2);

    // Configurar interrupciones
    int_init();
    int_connect(INT_TIMER0, speedCtrl);
    int_global_enable();
    
    // Timer cada 10ms para control más frecuente y suave
    timer_period(0, 0.01f);
}

void speedCtrl(void){
    int vD, vI;
    int uD, uI;
    int eD, eI;
    static int eDp=0, eIp=0, eDpp=0, eIpp=0;  // Para término derivativo
    static int integralD=0, integralI=0;      // Integradores separados
    static int uDp=0, uIp=0;
    
    // Filtro mejorado: promedio móvil de 8 muestras
    static int filtroD[8] = {0}, filtroI[8] = {0};
    static int indice = 0;
    int sumaD = 0, sumaI = 0;
    int j;
            
    // Leer velocidades actuales
    srv2_vel(vD, vI);
    
    // Filtro promedio móvil mejorado de 8 muestras
    filtroD[indice] = vD;
    filtroI[indice] = vI;
    
    for(j = 0; j < 8; j++){
        sumaD += filtroD[j];
        sumaI += filtroI[j];
    }
    
    velD = sumaD / 8;  // Velocidad filtrada motor derecho
    velI = sumaI / 8;  // Velocidad filtrada motor izquierdo
    
    indice = (indice + 1) % 8;  // Circular buffer
    
    // Calcular errores
    eD = Vdr - velD;
    eI = Vir - velI;
    
    // Acumular integrales con anti-windup
    integralD += eD;
    integralI += eI;
    
    // Límites del integrador para evitar windup
    if(integralD > MAX_INTEGRAL) integralD = MAX_INTEGRAL;
    else if(integralD < MIN_INTEGRAL) integralD = MIN_INTEGRAL;
    if(integralI > MAX_INTEGRAL) integralI = MAX_INTEGRAL;
    else if(integralI < MIN_INTEGRAL) integralI = MIN_INTEGRAL;
    
    // Controlador PID completo
    uD = VdKp*eD + VdKi*integralD + VdKd*(eD - eDp);
    uI = ViKp*eI + ViKi*integralI + ViKd*(eI - eIp);
    
    // Filtro de salida para suavizar PWM
    uD = (uD + uDp) / 2;
    uI = (uI + uIp) / 2;
    
    // Guardar valores anteriores
    eDpp = eDp;
    eIpp = eIp;
    eDp = eD;
    eIp = eI;
    uDp = uD;
    uIp = uI;
    
    // Limitar salida PWM con margen de seguridad
    if(uD < -1800) uD = -1800;
    else if(uD > 1800) uD = 1800;
    if(uI < -1800) uI = -1800;
    else if(uI > 1800) uI = 1800;
    
    // Aplicar PWM a los motores
    srv2_pwm(uD, uI);
}

void main(void){
    char str[256];
    static int boton_anterior = 0;
    static int motor_activo = 0;
    static int contador_rampa = 0;
    
    init();
    
    for(;;){
        // Detectar flanco de subida del botón 2
        if((_Btn & 2) && !(boton_anterior & 2)){
            if(!motor_activo){
                // Iniciar motores con rampa suave
                motor_activo = 1;
                vel_actual_D = 0;
                vel_actual_I = 0;
                timer_start(0);
                _Led = 1;
            }
        }
        
        // Detectar flanco de subida del botón 4
        if((_Btn & 4) && !(boton_anterior & 4)){
            // Parar motores inmediatamente
            motor_activo = 0;
            timer_stop(0);
            srv2_pwm(0, 0);
            Vdr = 0;
            Vir = 0;
            vel_actual_D = 0;
            vel_actual_I = 0;
            _Led = 0;
        }
        
        // Rampa suave de velocidad cuando los motores están activos
        if(motor_activo){
            contador_rampa++;
            
            // Incrementar velocidad gradualmente cada 50ms
            if(contador_rampa >= 5){  // 5 * 10ms = 50ms
                contador_rampa = 0;
                
                // Rampa de subida suave
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
                
                // Establecer velocidades objetivo suavemente
                Vdr = vel_actual_D;
                Vir = vel_actual_I;
            }
        }
        
        // Guardar estado anterior del botón
        boton_anterior = _Btn;
        
        // Opcional: Monitoreo por USB (descomenta para debug)
        /*
        if(motor_activo){
            sprintf(str, "Obj:%d VelD:%d VelI:%d PWM_D:%d PWM_I:%d\n", 
                    velocidad_objetivo, velD, velI, 
                    (Vdr != 0) ? 1 : 0, (Vir != 0) ? 1 : 0);
            usb_print(str);
        }
        */
        
        delay_ms(10);  // Loop cada 10ms para respuesta suave
    }
}