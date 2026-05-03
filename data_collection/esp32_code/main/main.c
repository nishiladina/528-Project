#include <stdio.h>
#include <sys/stat.h>
#include "i2cdev.h"
#include "mpu6050.h"
#include "esp_timer.h"

#define I2C_MASTER_SCL_IO 20
#define I2C_MASTER_SDA_IO 21
#define I2C_MASTER_NUM I2C_NUM_0
#define IMU_ADDR 0x68


void app_main(void)
{
    ESP_ERROR_CHECK(i2cdev_init());
    mpu6050_dev_t dev = {0};
    
    mpu6050_init_desc(&dev, IMU_ADDR, I2C_NUM_0,
                    I2C_MASTER_SDA_IO,
                    I2C_MASTER_SCL_IO);

    mpu6050_init(&dev);
    mpu6050_acceleration_t accel;
    mpu6050_rotation_t gyro;

    // loop wrapper: just repeat infinitely
    while (true) {
        // collect data 33 times
        for (int i = 1; i <= 33; i++){
            printf("FILE_START:%02d\n", i);
            printf("AX,AY,AZ,GX,GY,GZ\n");
            int64_t start_time = esp_timer_get_time();
            // Run for 3 seconds
            while (esp_timer_get_time() - start_time < 3000000)
            {
                mpu6050_get_acceleration(&dev, &accel);
                mpu6050_get_rotation(&dev, &gyro);

                printf("%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n", accel.x, accel.y, accel.z, gyro.x, gyro.y, gyro.z);
                
                // wait 30 ms before the next reading (33 hz)
                vTaskDelay(pdMS_TO_TICKS(30));
            }

            printf("FILE_END:%02d\n", i);
        }
    }
    

}
