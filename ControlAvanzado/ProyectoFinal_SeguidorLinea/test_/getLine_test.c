#include <qrt8.h>
#include <digio.h>
#include <rgb.h>
#include <uart.h>
#include <stdlib.h>
#include <xQuP01v0.h>


#define WHITE_THRESHOLD 200


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
    
    // Convertir a valores para cálculo (usa los valores raw divididos por 16)
    d0 = sensor_values[0]>>4;
    d1 = sensor_values[1]>>4;
    d2 = sensor_values[2]>>4;
    d3 = sensor_values[3]>>4;
    d4 = sensor_values[4]>>4;
    d5 = sensor_values[5]>>4;
    d6 = sensor_values[6]>>4;
    d7 = sensor_values[7]>>4;
    
    // Calcular posición ponderada (tu fórmula original)
    d = -320*d0-280*d1-240*d2-200*d3+200*d4+240*d5+280*d6+320*d7;
    m = d0+d1+d2+d3+d4+d5+d6+d7;
    
    if(m == 0) return 0;  // Sin línea detectada
    
    return (int)(d/m);    // Posición: negativo=izquierda, positivo=derecha
}


void main(void){
    int sensor_values[8];
    int led_pattern;
    int line_position;
    int i;
    char buffer[16];
    
    int bt_counter = 0;
    
    // Configurar UART para Bluetooth
    uart_setup(UART_B115200);


    qrt8_led_on(128); 
    
    uart_println("=== Test Deteccion Linea ===");
    
    for(;;){
    
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
        
        // Enviar datos por Bluetooth cada 50 ciclos
        bt_counter++;
        if(bt_counter >= 50){
            bt_counter = 0;
            
            // Enviar valores raw usando itoa10
            uart_print("Raw: ");
            for(i = 0; i < 8; i++){
                itoa10(sensor_values[i], buffer);
                uart_print(buffer);
                uart_print(" ");
            }
            uart_println("");
            
            // Enviar posición
            uart_print("Pos: ");
            itoa10(line_position, buffer);
            uart_println(buffer);
            
            // Estado simple
            if(line_position == 0){
                uart_println("SIN LINEA");
            } else if(line_position > 100){
                uart_println("MUY DERECHA");
            } else if(line_position > 50){
                uart_println("DERECHA");
            } else if(line_position < -100){
                uart_println("MUY IZQUIERDA");
            } else if(line_position < -50){
                uart_println("IZQUIERDA");
            } else {
                uart_println("CENTRO");
            }
            
            uart_println("---");
        }
        
        delay_ms(10);
    }
}