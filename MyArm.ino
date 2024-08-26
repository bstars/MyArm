#include "Wire.h"
#include "Adafruit_PWMServoDriver.h"


#define MIN_PULSE_WIDTH 500
#define MAX_PULSE_WIDTH 2500
#define FREQUENCY 50


Adafruit_PWMServoDriver pwm_driver;
float angles[4] = {0., 0., 0., 0.};
int servo_map[4] = {8, 9, 10, 11};
float servo_calibrate[4] = {2., -4., -10., 0.};


long float_map(float in, float low_in, float high_in, long low_out, long high_out){
    return low_out + (in - low_in) / (high_in - low_in) * (high_out - low_out);
}


void parse_float_array(float* arr, String s, size_t n){
    float f = 0.;
    for(size_t i=0; i<n; i++){
        f = s.toFloat();
        // Serial.println(f);
        arr[i] = f;
        s = s.substring(s.indexOf(' ') + 1);
    }
}


void write_degrees(){
    long pulsewidth = 0;
    for(int i=0; i<=3; i++){
        pulsewidth = float_map(angles[i] - servo_calibrate[i], -90, 90, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH);
        pulsewidth = min(MAX_PULSE_WIDTH, max(pulsewidth, MIN_PULSE_WIDTH));
        pwm_driver.writeMicroseconds(servo_map[i], pulsewidth);
    }
}




void setup(){
    Serial.begin(9600);
    Serial.setTimeout(20);
    pwm_driver.begin();
    pwm_driver.setPWMFreq(FREQUENCY);
    write_degrees();
}

void loop(){
    if (Serial.available() > 0){
        String msg = Serial.readString();
        parse_float_array(angles, msg, 4);
        write_degrees();
        Serial.println("done");
    }
}