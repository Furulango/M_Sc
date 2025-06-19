#include <qrt8.h>
#include <digio.h>
#include <rgb.h>
#include <uart.h>
#include <stdlib.h>
#include <servo2d.h>
#include <interrupt.h>
#include <xQuP01v0.h>

#define WHITE_THRESHOLD 200

// Parámetros del controlador PID (tu controlador)
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
int velocidad_objetivo = 15;  

// Variables para rampa suave de velocidad
int vel_actual_D = 0, vel_actual_I = 0;
int incremento_rampa = 1;  

void speedCtrl(void);

void init(void){
    // Habilitar PWMs
    srv2_enable(srv2AXIS1|srv2AXIS2);

    // Configurar interrupciones
    int_init();
    int_connect(INT_TIMER0, speedCtrl);
    int_global_enable();
    
    // Timer cada 10ms
    timer_period(0, 0.01f);
}

int getLine(void){
    float d, m;
    float d0,d1,d2,d3,d4,d5,d6,d7;
    int sensor_values[8];

    qrt8_read();

    // Mapeo igual que tu código
    sensor_values[0] = qrt8_sensor(7);  // Sensor izquierda
    sensor_values[1] = qrt8_sensor(6);
    sensor_values[2] = qrt8_sensor(5);
    sensor_values[3] = qrt8_sensor(4);
    sensor_values[4] = qrt8_sensor(3);
    sensor_values[5] = qrt8_sensor(2);
    sensor_values[6] = qrt8_sensor(1);
    sensor_values[7] = qrt8_sensor(0);  // Sensor derecha
    
    // Convertir a valores para cálculo
    d0 = sensor_values[0]>>4;
    d1 = sensor_values[1]>>4;
    d2 = sensor_values[2]>>4;
    d3 = sensor_values[3]>>4;
    d4 = sensor_values[4]>>4;
    d5 = sensor_values[5]>>4;
    d6 = sensor_values[6]>>4;
    d7 = sensor_values[7]>>4;
    
    // Calcular posición ponderada
    d = -320*d0-280*d1-240*d2-200*d3+200*d4+240*d5+280*d6+320*d7;
    m = d0+d1+d2+d3+d4+d5+d6+d7;
    
    if(m == 0) return 0;  // Sin línea detectada
    
    return (int)(d/m);    // Posición: negativo=izquierda, positivo=derecha
}

void speedCtrl(void){
    int vD, vI;
    int uD, uI;
    int eD, eI;
    static int eDp=0, eIp=0, eDpp=0, eIpp=0;  
    static int integralD=0, integralI=0;      
    static int uDp=0, uIp=0;
    
    // Si las velocidades objetivo son 0, apagar motores y resetear PID
    if(Vdr == 0 && Vir == 0){
        srv2_pwm(0, 0);
        // Reset variables PID para evitar acumulación
        eDp = 0; eIp = 0; eDpp = 0; eIpp = 0;
        integralD = 0; integralI = 0;
        uDp = 0; uIp = 0;
        return;
    }
    
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
    
    // Limitar salida PWM
    if(uD < -1800) uD = -1800;
    else if(uD > 1800) uD = 1800;
    if(uI < -1800) uI = -1800;
    else if(uI > 1800) uI = 1800;
    
    // Aplicar PWM a los motores
    srv2_pwm(uD, uI);
}

void main(void){
    int sensor_values[8];
    int led_pattern;
    int line_position;
    int i;
    char buffer[16];
    
    int bt_counter = 0;
    static int boton_anterior = 0;
    static int motor_activo = 0;
    static int contador_rampa = 0;
    
    // Configurar UART para Bluetooth
    uart_setup(UART_B115200);
    qrt8_led_on(128); 
    
    init();
    
    uart_println("=== Seguidor Centro Linea ===");
    uart_println("Boton 2: Activar/Desactivar");
    uart_println("Solo avanza cuando detecta CENTRO");
    
    for(;;){
        qrt8_read();
        
        // Tu mapeo original
        sensor_values[0] = qrt8_sensor(7);  // Sensor izquierda
        sensor_values[1] = qrt8_sensor(6);
        sensor_values[2] = qrt8_sensor(5);
        sensor_values[3] = qrt8_sensor(4);
        sensor_values[4] = qrt8_sensor(3);
        sensor_values[5] = qrt8_sensor(2);
        sensor_values[6] = qrt8_sensor(1);
        sensor_values[7] = qrt8_sensor(0);  // Sensor derecha
        
        // Tu lógica de detección original para LEDs
        led_pattern = 0;
        for(i = 0; i < 8; i++){
            if(sensor_values[i] > WHITE_THRESHOLD){
                led_pattern |= (1 << i); 
            }
        }
        
        _Led = led_pattern;
        
        // Calcular posición ponderada
        line_position = getLine();
        
        // Verificar si hay línea detectada (al menos un sensor activo)
        int linea_detectada = 0;
        for(i = 0; i < 8; i++){
            if(sensor_values[i] > WHITE_THRESHOLD){
                linea_detectada = 1;
                break;
            }
        }
        
        // Control de botones
        if((_Btn & 2) && !(boton_anterior & 2)){
            if(!motor_activo){
                motor_activo = 1;
                vel_actual_D = 0;
                vel_actual_I = 0;
                timer_start(0);
                uart_println("ROBOT ACTIVADO");
                rgb_color(rgbGREEN);
            } else {
                motor_activo = 0;
                timer_stop(0);
                srv2_pwm(0, 0);
                Vdr = 0;
                Vir = 0;
                vel_actual_D = 0;
                vel_actual_I = 0;
                uart_println("ROBOT DESACTIVADO");
                rgb_color(rgbRED);
            }
        }
        
        // Parada de emergencia
        if((_Btn & 4) && !(boton_anterior & 4)){
            motor_activo = 0;
            timer_stop(0);
            srv2_pwm(0, 0);
            Vdr = 0;
            Vir = 0;
            vel_actual_D = 0;
            vel_actual_I = 0;
            uart_println("PARADA DE EMERGENCIA");
            rgb_color(rgbRED);
        }
        
        // LÓGICA PRINCIPAL: Solo avanzar si está activo Y hay línea Y está en el CENTRO
        if(motor_activo && linea_detectada && line_position >= -50 && line_position <= 50){
            // Rampa suave de velocidad cuando detecta centro
            contador_rampa++;
            
            if(contador_rampa >= 5){  // Cada 50ms
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
                
                // Establecer velocidades objetivo (negativas para avanzar con bola loca adelante)
                Vdr = -vel_actual_D;
                Vir = -vel_actual_I;
            }
            rgb_color(rgbGREEN);  // Verde cuando avanza
        } else {
            // No está en el centro o no está activo: parar
            Vdr = 0;
            Vir = 0;
            vel_actual_D = 0;
            vel_actual_I = 0;
            
            if(motor_activo){
                if(!linea_detectada){
                    rgb_color(rgbRED);    // Sin línea = Rojo
                } else {
                    rgb_color(rgbBLUE);   // Línea fuera del centro = Azul
                }
            }
        }
        
        boton_anterior = _Btn;
        
        // Enviar datos por Bluetooth cada 50 ciclos
        bt_counter++;
        if(bt_counter >= 50){
            bt_counter = 0;
            
            // Enviar posición
            uart_print("Pos: ");
            itoa10(line_position, buffer);
            uart_println(buffer);
            
            // Estado simple
            if(!motor_activo){
                uart_println("DESACTIVADO");
            } else if(!linea_detectada){
                uart_println("SIN LINEA - PARADO");
            } else if(line_position >= -50 && line_position <= 50){
                uart_println("CENTRO - AVANZANDO");
            } else {
                uart_println("FUERA CENTRO - PARADO");
            }
            
            uart_println("---");
        }
        
        delay_ms(10);
    }
}