#include <Arduino.h>
#include <SPI.h>
#include "ICM42688.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

extern "C" {
  #include "NetworkController.h"
  #include "functional.h"
  #include "param/test_controller_fwr_conf.h"
}

// ================= Serial =================
#define COMMUNICATION_SERIAL Serial
#define COMMUNICATION_SERIAL_BAUD 115200
static const uint8_t START_BYTE_SERIAL_FWR = 0x9B;

// keep binary stream clean. Only print text when binary is OFF.
static volatile bool g_binary_enabled = false;

SemaphoreHandle_t serialMutex;

// ================= IMU (SPI) =================
#define MOSI_PIN      11
#define MISO_PIN      14
#define SCLK_PIN      12
#define CSN_IMU_PIN   10
#define IMU_INT_PIN   16

ICM42688 IMU(SPI, CSN_IMU_PIN);

// ================= Network =================
NetworkController controller;
static float inputs[13] = {0};

TaskHandle_t network_task_handle = NULL;
TaskHandle_t serial_task_handle  = NULL;

// ================= ORIGINAL binary protocol =================
struct __attribute__((packed)) serial_control_out {
  uint16_t sequence_num;

  float pitch_offset;

  float attitude_pitch;
  float attitude_roll;
  float attitude_yaw_rate;

  float raw_gyro_x;
  float raw_gyro_y;
  float raw_gyro_z;
  float raw_acc_x;
  float raw_acc_y;
  float raw_acc_z;

  uint8_t checksum_out;
};

static inline uint8_t calc_checksum(const serial_control_out& pkt) {
  const uint8_t* p = (const uint8_t*)&pkt;
  uint8_t sum = 0;
  for (uint16_t i = 0; i < sizeof(serial_control_out) - 1; i++) {
    sum += p[i];
  }
  return sum;
}

// ================= Timing stats =================
static volatile uint32_t inf_us_last = 0;
static volatile uint64_t inf_us_sum  = 0;
static volatile uint32_t inf_count   = 0;
static volatile uint32_t inf_us_max  = 0;
static uint32_t status_t0_us = 0;

// ================= Spike sparsity stats (per inference) =================
static volatile uint16_t enc_k_last  = 0;
static volatile uint16_t hid_k_last  = 0;
static volatile uint16_t hid2_k_last = 0;
static volatile uint32_t enc_k_sum   = 0;
static volatile uint32_t hid_k_sum   = 0;
static volatile uint32_t hid2_k_sum  = 0;

// ================= Utils =================
static void safePrint(const char* fmt, ...) {
  if (xSemaphoreTake(serialMutex, pdMS_TO_TICKS(10)) == pdTRUE) {
    va_list args;
    va_start(args, fmt);
    char buf[192];
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    COMMUNICATION_SERIAL.print(buf);
    xSemaphoreGive(serialMutex);
  }
}

static void sendBinaryOriginal(const serial_control_out& pkt) {
  if (!g_binary_enabled) return;
  if (xSemaphoreTake(serialMutex, 0) == pdTRUE) {
    COMMUNICATION_SERIAL.write(START_BYTE_SERIAL_FWR);
    COMMUNICATION_SERIAL.write((const uint8_t*)&pkt, sizeof(serial_control_out));
    xSemaphoreGive(serialMutex);
  }
}

// ================= ISR =================
void IRAM_ATTR imuDataReadyISR() {
  BaseType_t hpTaskWoken = pdFALSE;
  if (network_task_handle) {
    vTaskNotifyGiveFromISR(network_task_handle, &hpTaskWoken);
  }
  if (hpTaskWoken) portYIELD_FROM_ISR();
}

// ================= Serial command task =================
static void serial_task(void* pv) {
  String line;
  while (1) {
    while (COMMUNICATION_SERIAL.available()) {
      char c = (char)COMMUNICATION_SERIAL.read();
      if (c == '\n' || c == '\r') {
        line.trim();
        line.toLowerCase();
        if (line.length() > 0) {
          if (line == "enable") {
            // Enable binary output (silent to keep stream clean)
            g_binary_enabled = true;
          } else if (line == "disable") {
            g_binary_enabled = false;
            safePrint("[cmd] binary disabled\n");
          } else {
            if (!g_binary_enabled) safePrint("[cmd] unknown: %s\n", line.c_str());
          }
        }
        line = "";
      } else {
        if (line.length() < 64) line += c;
      }
    }
    vTaskDelay(pdMS_TO_TICKS(5));
  }
}

// ================= Network task =================
static void network_task(void* pv) {
  serial_control_out out_pkt;

  while (ulTaskNotifyTake(pdTRUE, portMAX_DELAY)) {
    // 1) Read IMU
    IMU.getAGT();
    float gx = IMU.gyrX();
    float gy = IMU.gyrY();
    float gz = IMU.gyrZ();
    float ax = IMU.accX();
    float ay = IMU.accY();
    float az = IMU.accZ();

    // 2) Fill inputs
    inputs[0] = gx; inputs[1] = gy; inputs[2] = gz;
    inputs[3] = ax; inputs[4] = ay; inputs[5] = az;

    inputs[6] = 0.0f; // target_pitch

    inputs[7]  = gx;
    inputs[8]  = gy;
    inputs[9]  = gz;
    inputs[10] = 0.0f;
    inputs[11] = 0.0f;
    inputs[12] = 0.0f;

    set_network_input(&controller, inputs);

    // 3) Inference timing (ONLY forward_network)
    uint32_t t0 = micros();
    forward_network(&controller);
    uint32_t dt = micros() - t0;

    inf_us_last = dt;
    inf_us_sum += dt;
    inf_count++;
    if (dt > inf_us_max) inf_us_max = dt;

    // 3b) Spike counts (k/N) for sparsity statistics
    enc_k_last  = controller.enc->spike_k;
    hid_k_last  = controller.hid->spike_k;
    hid2_k_last = controller.hid2->spike_k;
    enc_k_sum  += enc_k_last;
    hid_k_sum  += hid_k_last;
    hid2_k_sum += hid2_k_last;

    // 4) Build ORIGINAL protocol packet (55B payload + checksum)
    float* att = get_attitude_outputs(&controller);

    out_pkt.sequence_num = 0;
    out_pkt.pitch_offset = controller.out[0];

    out_pkt.attitude_pitch    = att[0];
    out_pkt.attitude_roll     = att[1];
    out_pkt.attitude_yaw_rate = att[2];

    out_pkt.raw_gyro_x = gx;
    out_pkt.raw_gyro_y = gy;
    out_pkt.raw_gyro_z = gz;
    out_pkt.raw_acc_x  = ax;
    out_pkt.raw_acc_y  = ay;
    out_pkt.raw_acc_z  = az;

    out_pkt.checksum_out = calc_checksum(out_pkt);

    sendBinaryOriginal(out_pkt);
  }
}

void setup() {
  serialMutex = xSemaphoreCreateMutex();
  COMMUNICATION_SERIAL.begin(COMMUNICATION_SERIAL_BAUD);
  delay(300);

  safePrint("SNN minimal (original binary protocol). Type enable/disable.\n");

  SPI.begin(SCLK_PIN, MISO_PIN, MOSI_PIN, CSN_IMU_PIN);
  int st = IMU.begin();
  if (st < 0) {
    safePrint("IMU init failed: %d\n", st);
    while (1) {}
  }
  IMU.setAccelODR(ICM42688::odr100);
  IMU.setGyroODR(ICM42688::odr100);

  pinMode(IMU_INT_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(IMU_INT_PIN), imuDataReadyISR, RISING);
  IMU.enableDataReadyInterrupt();

  safePrint("Building controller...\n");
  controller = build_network(6, 150, 150, 130, 1);
  init_network(&controller);
  load_network_from_header(&controller, &conf);
  reset_network(&controller);

  xTaskCreatePinnedToCore(network_task, "NetTask", 8192, NULL, 3, &network_task_handle, 1);
  xTaskCreatePinnedToCore(serial_task,  "SerTask", 4096, NULL, 1, &serial_task_handle,  0);

  status_t0_us = micros();
}

void loop() {
  if (micros() - status_t0_us >= 1000000UL) {
    uint32_t cnt  = inf_count;
    uint64_t sum  = inf_us_sum;
    uint32_t mx   = inf_us_max;
    uint32_t last = inf_us_last;

    float avg_us = (cnt > 0) ? (float)sum / (float)cnt : 0.0f;

    float enc_k_avg  = (cnt > 0) ? (float)enc_k_sum  / (float)cnt : 0.0f;
    float hid_k_avg  = (cnt > 0) ? (float)hid_k_sum  / (float)cnt : 0.0f;
    float hid2_k_avg = (cnt > 0) ? (float)hid2_k_sum / (float)cnt : 0.0f;

    // IMPORTANT: keep binary stream clean. Only print text when binary is OFF.
    if (!g_binary_enabled) {
      safePrint("infer_us: avg=%.2f last=%u max=%u n=%u | enc_k: avg=%.1f last=%u/%d | hid_k: avg=%.1f last=%u/%d | hid2_k: avg=%.1f last=%u/%d\n",
                avg_us, last, mx, cnt,
                enc_k_avg,  (unsigned)enc_k_last,  controller.enc->size,
                hid_k_avg,  (unsigned)hid_k_last,  controller.hid->size,
                hid2_k_avg, (unsigned)hid2_k_last, controller.hid2->size);
    }

    inf_us_sum = 0;
    inf_count  = 0;
    inf_us_max = 0;

    enc_k_sum  = 0;
    hid_k_sum  = 0;
    hid2_k_sum = 0;

    status_t0_us = micros();
  }

  delay(5);
}