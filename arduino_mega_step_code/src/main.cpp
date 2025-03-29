#include <Arduino.h>
#include <TMCStepper.h>
#include <AccelStepper.h>

// TMC2209 Configuration
#define L_SERIAL Serial3  // Left motor
#define R_SERIAL Serial2  // Right motor
#define R_SENSE 0.11f

TMC2209Stepper driverL(&L_SERIAL, R_SENSE, 0x00);
TMC2209Stepper driverR(&R_SERIAL, R_SENSE, 0x00);

// Stepper Pins
#define R_STEP_PIN 54
#define R_DIR_PIN 55
#define R_ENABLE_PIN 38

#define L_STEP_PIN 26
#define L_DIR_PIN 28
#define L_ENABLE_PIN 24

// Stepper Objects
AccelStepper stepperL(AccelStepper::DRIVER, L_STEP_PIN, L_DIR_PIN);
AccelStepper stepperR(AccelStepper::DRIVER, R_STEP_PIN, R_DIR_PIN);

// Constants
const unsigned long REPORT_INTERVAL = 100;  // Report position every 100ms
const unsigned long COMMAND_TIMEOUT = 1000; // Stop if no command received in 1s
unsigned long last_report = 0;
unsigned long last_command = 0;

// Debug flags
const bool DEBUG_COMMANDS = true;  // Print received commands

// Current speed targets
float target_speed_left = 0.0;   // Steps per second
float target_speed_right = 0.0;  // Steps per second
bool motors_running = false;     // Flag to track if motors are active

void updateMotorSpeeds();

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000);  // Wait for serial but with timeout
  Serial.println("MEGA: Stepper velocity control ready");
  
  // TMC2209 Initialization
  L_SERIAL.begin(115200);
  R_SERIAL.begin(115200);
  
  driverL.begin();
  driverL.pdn_disable(true);
  driverL.toff(5);
  driverL.en_spreadCycle(false);
  driverL.microsteps(16);
  driverL.irun(31);
  driverL.ihold(5);

  driverR.begin();
  driverR.pdn_disable(true);
  driverR.toff(5);
  driverR.en_spreadCycle(false);
  driverR.microsteps(16);
  driverR.irun(31);
  driverR.ihold(5);

  // Stepper Configuration
  stepperL.setEnablePin(L_ENABLE_PIN);
  stepperL.setPinsInverted(true, false, true);
  stepperL.enableOutputs();
  stepperL.setMaxSpeed(40000);  // Max steps/second
  stepperL.setAcceleration(40000);  // Steps/second²

  stepperR.setEnablePin(R_ENABLE_PIN);
  stepperR.setPinsInverted(false, false, true);
  stepperR.enableOutputs();
  stepperR.setMaxSpeed(40000);
  stepperR.setAcceleration(40000);
  
  // Start with motors stopped
  stepperL.setSpeed(0);
  stepperR.setSpeed(0);
}

void loop() {
  // Check for command timeout (safety feature)
  if (millis() - last_command > COMMAND_TIMEOUT && motors_running) {
    // No recent commands, stop motors
    if (DEBUG_COMMANDS) {
      Serial.println("MEGA: Command timeout, stopping motors");
    }
    target_speed_left = 0;
    target_speed_right = 0;
    updateMotorSpeeds();
    motors_running = false;
  }

  // Handle incoming commands
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    
    if (cmd.startsWith("VEL:")) {  // Velocity command format
      cmd.remove(0, 4);
      int comma = cmd.indexOf(',');
      if (comma != -1) {
        float left = cmd.substring(0, comma).toFloat();
        float right = cmd.substring(comma+1).toFloat();
        
        // Check if speeds are too low but non-zero
        if (abs(left) > 0 && abs(left) < 5) left = (left < 0) ? -5 : 5;
        if (abs(right) > 0 && abs(right) < 5) right = (right < 0) ? -5 : 5;
        
        // Update target speeds
        target_speed_left = left;
        target_speed_right = right;
        
        // Debug output
        if (DEBUG_COMMANDS) {
          Serial.print("MEGA: Speed set to L=");
          Serial.print(target_speed_left);
          Serial.print(", R=");
          Serial.println(target_speed_right);
        }
        
        // Update motor speeds
        updateMotorSpeeds();
        
        // Reset command timeout
        last_command = millis();
      }
    }
    else if (cmd.length() > 0) {
      // Unknown command
      Serial.print("MEGA: Unknown command: ");
      Serial.println(cmd);
    }
  }

  // Run motors at their set speeds
  if (motors_running) {
    stepperL.runSpeed();  // This steps the motor one step when needed to maintain speed
    stepperR.runSpeed();
  }

  // Report positions
  if (millis() - last_report > REPORT_INTERVAL) {
    Serial.print("STEPS:");
    Serial.print(stepperL.currentPosition());
    Serial.print(",");
    Serial.println(stepperR.currentPosition());
    last_report = millis();
  }
}

// Set motor speeds based on the target values
void updateMotorSpeeds() {
  // Set the constant speeds
  stepperL.setSpeed(target_speed_left);
  stepperR.setSpeed(target_speed_right);
  
  // Update running state based on whether either motor should be moving
  motors_running = (abs(target_speed_left) > 0 || abs(target_speed_right) > 0);
  
  if (DEBUG_COMMANDS) {
    if (motors_running) {
      Serial.println("MEGA: Motors running");
    } else {
      Serial.println("MEGA: Motors stopped");
    }
  }
}