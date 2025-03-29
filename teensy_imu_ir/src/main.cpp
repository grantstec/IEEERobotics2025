#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>
#include <MLX90640_API.h>
#include <MLX90640_I2C_Driver.h>

// Configuration
#define BNO055_I2C_ADDRESS 0x28  // Default I2C address
#define MLX90640_I2C_ADDRESS 0x33
#define IMU_UPDATE_RATE 50       // Hz (for IMU)
#define THERMAL_UPDATE_RATE 2    // Reduced to 2Hz to make it more reliable

#define TA_SHIFT 8
float mlx90640To[768];
paramsMLX90640 mlx90640;

// Use Wire for both sensors (we'll manage timing carefully)
Adafruit_BNO055 bno = Adafruit_BNO055(55, BNO055_I2C_ADDRESS);

// Debug flags
bool thermalEnabled = true;  // Set to false to disable thermal camera if needed
unsigned long lastSuccessfulThermal = 0;

// Forward declarations
bool initThermalCamera();
bool readThermalFrame();

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000); // Wait up to 3 seconds for serial
  
  Serial.println("Starting setup...");
  
  // Initialize I2C
  Wire.begin();
  Wire.setClock(100000); // Slow down to 100kHz for reliability
  
  delay(100); // Short delay
  
  // BNO055 setup
  if (!bno.begin()) {
    Serial.println("BNO055 NOT DETECTED! Check wiring.");
    while (1); // Halt if init fails
  }
  
  Serial.println("BNO055 initialized");
  delay(100);
  
  bno.setExtCrystalUse(true);
  bno.setMode(OPERATION_MODE_NDOF);  // 9-DOF fusion mode
  
  delay(100); // Short delay
  
  // Initialize thermal camera
  if (thermalEnabled) {
    if (!initThermalCamera()) {
      Serial.println("WARNING: Thermal camera initialization failed!");
      Serial.println("Continuing with IMU only.");
      thermalEnabled = false;
    } else {
      Serial.println("Thermal camera initialized");
    }
  }
  
  Serial.println("Setup complete, starting measurement loop...");
}

bool initThermalCamera() {
  // Check if MLX90640 is connected
  Wire.beginTransmission(MLX90640_I2C_ADDRESS);
  byte error = Wire.endTransmission();
  if (error != 0) {
    Serial.print("MLX90640 not found, I2C error: ");
    Serial.println(error);
    return false;
  }
  
  // Initialize MLX90640
  uint16_t eeMLX90640[832];
  
  // Try to read EEPROM up to 3 times
  int status = -1;
  for (int attempt = 0; attempt < 3; attempt++) {
    Serial.print("Attempting to read MLX90640 EEPROM (attempt ");
    Serial.print(attempt + 1);
    Serial.println(")");
    
    status = MLX90640_DumpEE(MLX90640_I2C_ADDRESS, eeMLX90640);
    if (status == 0) break;
    
    delay(100);
  }
  
  if (status != 0) {
    Serial.print("Failed to read MLX90640 EEPROM: ");
    Serial.println(status);
    return false;
  }
  
  status = MLX90640_ExtractParameters(eeMLX90640, &mlx90640);
  if (status != 0) {
    Serial.print("Failed to extract MLX90640 parameters: ");
    Serial.println(status);
    return false;
  }

  // Set thermal camera to lowest refresh rate for reliability
  status = MLX90640_SetRefreshRate(MLX90640_I2C_ADDRESS, 0x02); // 2Hz
  if (status != 0) {
    Serial.print("Refresh rate setting failed: ");
    Serial.println(status);
    // Continue anyway
  }
  
  status = MLX90640_SetResolution(MLX90640_I2C_ADDRESS, 0x03);  // 19-bit resolution
  if (status != 0) {
    Serial.print("Resolution setting failed: ");
    Serial.println(status);
    // Continue anyway
  }
  
  lastSuccessfulThermal = millis();
  return true;
}

void loop() {
  static unsigned long lastImuUpdate = 0;
  static unsigned long lastThermalUpdate = 0;
  unsigned long now = millis();

  // IMU update (50Hz)
  if (now - lastImuUpdate >= (1000/IMU_UPDATE_RATE)) {
    lastImuUpdate = now;
    
    // Get all sensor data
    imu::Quaternion quat = bno.getQuat();
    imu::Vector<3> gyro = bno.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);
    imu::Vector<3> accel = bno.getVector(Adafruit_BNO055::VECTOR_ACCELEROMETER);

    // Send formatted IMU data
    Serial.print("IMU:");
    Serial.print(quat.x(), 4); Serial.print(",");
    Serial.print(quat.y(), 4); Serial.print(",");
    Serial.print(quat.z(), 4); Serial.print(",");
    Serial.print(quat.w(), 4); Serial.print(",");
    Serial.print(gyro.x(), 4); Serial.print(",");
    Serial.print(gyro.y(), 4); Serial.print(",");
    Serial.print(gyro.z(), 4); Serial.print(",");
    Serial.print(accel.x(), 4); Serial.print(",");
    Serial.print(accel.y(), 4); Serial.print(",");
    Serial.println(accel.z(), 4);
  }

  // Thermal camera update (slower rate)
  if (thermalEnabled && now - lastThermalUpdate >= (1000/THERMAL_UPDATE_RATE)) {
    lastThermalUpdate = now;
    
    if (!readThermalFrame()) {
      // Check if thermal camera has been failing for too long (30 seconds)
      if (now - lastSuccessfulThermal > 30000) {
        Serial.println("Thermal camera not responding for 30 seconds, attempting to reinitialize...");
        
        // Try to reinitialize
        if (!initThermalCamera()) {
          Serial.println("Thermal camera reinitialization failed, disabling thermal camera.");
          thermalEnabled = false;
        } else {
          Serial.println("Thermal camera reinitialized successfully.");
        }
      }
    }
  }
  
  // Small delay to ensure we don't overrun the serial buffer
  // and to give other tasks time to run
  delay(1);
}

bool readThermalFrame() {
  // Clear the I2C bus before starting
  Wire.setClock(100000); // Slow down for thermal camera
  
  uint16_t mlx90640Frame[834];
  int status = MLX90640_GetFrameData(MLX90640_I2C_ADDRESS, mlx90640Frame);
  
  if (status == 0) {  // Frame data retrieved successfully
    lastSuccessfulThermal = millis();
    
    float Ta = MLX90640_GetTa(mlx90640Frame, &mlx90640);
    MLX90640_CalculateTo(mlx90640Frame, &mlx90640, 0.95, Ta - TA_SHIFT, mlx90640To);
    
    // Print thermal data    
    bool fireDetected = false;
    for (int i = 0; i < 768; i++) {
      if (mlx90640To[i] > 40.0) {
        fireDetected = true;
      }
    }
    
    Serial.print("FIRE:");
    Serial.println(fireDetected ? "1" : "0");
    
    return true;
  } else {
    return false;
  }
}