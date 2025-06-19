#include <qrt8.h>
#include <digio.h>
#include <rgb.h>
#include <xQuP01v0.h>
#define WHITE_THRESHOLD 200

void main(void){
    int sensor_values[8];
    int led_pattern;
    int i;
    
    qrt8_led_on(128); 
    
    for(;;){
        
        qrt8_read();
        
        // Mapeo 
        sensor_values[0] = qrt8_sensor(7);  // Sensor izquierda
        sensor_values[1] = qrt8_sensor(6);
        sensor_values[2] = qrt8_sensor(5);
        sensor_values[3] = qrt8_sensor(4);
        sensor_values[4] = qrt8_sensor(3);
        sensor_values[5] = qrt8_sensor(2);
        sensor_values[6] = qrt8_sensor(1);
        sensor_values[7] = qrt8_sensor(0);  // Sensor derecha
        
        led_pattern = 0;
        for(i = 0; i < 8; i++){
            if(sensor_values[i] > WHITE_THRESHOLD){
                led_pattern |= (1 << i); 
            }
        }
        
        _Led = led_pattern;
        
        delay_ms(10);
    }
}