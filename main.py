import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://a-iinterviewer.5316eictlws-2.workers.dev/"],  # 可加上你的正式前端網址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImprovedFeatureExtractorLibrosa:
    def __init__(self,
                 sr: int = 16_000,
                 n_fft: int = 512,
                 hop: int = 256,
                 n_mels: int = 64):
        self.sr = sr
        self.n_fft = n_fft
        self.hop = hop
        self.n_mels = n_mels

    # ------- 公開方法 -------
    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform: Tensor [1, T] or [T]，輸出 Tensor [3, 64, time]"""
        if waveform.ndim == 2:                       # [1, T] → [T]
            waveform = waveform[0]
        y = waveform.cpu().numpy()

        # Mel power → dB
        mel = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_fft=self.n_fft,
            hop_length=self.hop, n_mels=self.n_mels,
            fmin=20, fmax=self.sr // 2
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)

        # 全域標準化
        x = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
        
        T = x.shape[1]
        if T >= 9:
            delta1 = librosa.feature.delta(x)
            delta2 = librosa.feature.delta(x, order=2)
        else:
            # 時間框太短：改回傳 0 矩陣，避免 width 錯誤
            delta1 = np.zeros_like(x)
            delta2 = np.zeros_like(x)
        feat = np.stack([x, delta1, delta2], axis=0) # [3, 64, time]

        return torch.tensor(feat, dtype=torch.float32)
    
label2id = {
    'dep': 0, 'dre': 1, 'gar': 2, 'wah': 3, 'nem': 4, 'sep': 5, 'sui': 6,
    'lan': 7, 'bau': 8, 'nep': 9, 'tep': 10, 'wei': 11, 'rei': 12, 'pei': 13,
    'yeah': 14, 'ga': 15, 'aya': 16, 'bo': 17, 'kin': 18, 'het': 19, 'nen': 20,
    'vot': 21, 'tim': 22, 'qei': 23, 'lun': 24, 'tig': 25, 'zep': 26, 'wib': 27,
    'hon': 28, 'tud': 29, 'lut': 30, 'san': 31, 'yui': 32, 'ni': 33, 'cho': 34
}
id2label = {v: k for k, v in label2id.items()}

class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, down=False):
        super().__init__()
        stride = 2 if down else 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, 1, 1),
            nn.BatchNorm2d(out_c)
        )
        self.skip = nn.Conv2d(in_c, out_c, 1, stride) if in_c != out_c or down else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x) + self.skip(x))

class ResNetSmall(nn.Module):
    def __init__(self, n_class=35):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.layer1 = BasicBlock(32, 64, down=True)
        self.layer2 = BasicBlock(64, 128, down=True)
        self.layer3 = BasicBlock(128, 256, down=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, n_class)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        return self.fc(x)

# ──────────────────────────── 常數設定 ────────────────────────────
CKPT_PATH = "final_model.pth"  # 權重檔
SR_TARGET = 16_000                   # 目標取樣率
SEG_DUR_SEC = 2.0                  # 每段 2 秒 (固定窗)
MAX_LEN = 400                        # 與訓練一致的 frame 數

if torch.backends.mps.is_available():
    DEVICE: str = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

model = ResNetSmall(n_class=len(label2id)).to(DEVICE)
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model.eval()

extractor = ImprovedFeatureExtractorLibrosa()

# ────────────────────────────── 主流程 ──────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 讀取檔案內容
        wav_bytes = await file.read()
        
        # 檢查檔案內容
        print(f"接收到的檔案大小: {len(wav_bytes)} bytes")
        print(f"檔案類型: {file.content_type}")
        print(f"檔案名稱: {file.filename}")
        
        if len(wav_bytes) == 0:
            return JSONResponse(status_code=400, content={"error": "Empty audio file"})
        
        # 使用 tempfile 處理音訊檔案
        # 根據實際格式決定副檔名
        if file.content_type and 'webm' in file.content_type:
            suffix = '.webm'
        elif file.content_type and 'mp4' in file.content_type:
            suffix = '.mp4'
        elif file.content_type and 'ogg' in file.content_type:
            suffix = '.ogg'
        else:
            suffix = '.wav'  # 預設
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(wav_bytes)
            temp_file_path = temp_file.name
        
        try:
            # 使用 librosa 直接載入（支援多種格式）
            print(f"嘗試載入音訊檔案: {temp_file_path}")
            y, sr = librosa.load(temp_file_path, sr=None)
            print(f"載入成功，取樣率: {sr}, 長度: {len(y)}")
            
            # 確保是單聲道
            if y.ndim > 1:
                y = y.mean(axis=1)
                
            if sr != SR_TARGET:
                y = librosa.resample(y, orig_sr=sr, target_sr=SR_TARGET)
                sr = SR_TARGET

            # 檢查音訊長度
            if len(y) == 0:
                return JSONResponse(status_code=400, content={"error": "Empty audio data after processing"})

            # 切段
            seg_len = int(SEG_DUR_SEC * sr)
            segments = []
            for start in range(0, len(y), seg_len):
                end = start + seg_len
                if end <= len(y):
                    wav_arr = y[start:end]
                else:
                    wav_arr = np.pad(y[start:], (0, end - len(y)))
                segments.append(wav_arr)

            print(f"切分成 {len(segments)} 個片段")

            # 檢查是否有段落
            if not segments:
                return JSONResponse(status_code=400, content={"error": "No audio segments found"})

            # 特徵擷取
            feat_list = []
            for i, wav_arr in enumerate(segments):
                if len(wav_arr) == 0:
                    continue
                try:
                    wav_tensor = torch.from_numpy(wav_arr).unsqueeze(0)
                    feat = extractor.extract(wav_tensor)
                    T = feat.shape[2]
                    feat = F.pad(feat, (0, MAX_LEN - T)) if T < MAX_LEN else feat[:, :, :MAX_LEN]
                    feat_list.append(feat)
                except Exception as e:
                    print(f"特徵擷取失敗 (片段 {i}): {e}")
                    continue
            
            if not feat_list:
                return JSONResponse(status_code=400, content={"error": "No valid features extracted"})
                
            inputs = torch.stack(feat_list).to(DEVICE)
            print(f"特徵張量形狀: {inputs.shape}")

            # 推論
            with torch.no_grad():
                logits = model(inputs)
            preds = logits.argmax(1).cpu().tolist()
            print(f"預測結果: {preds}")

            # 回傳
            result = [
                {"start": round(i * SEG_DUR_SEC, 2),
                 "end": round((i + 1) * SEG_DUR_SEC, 2),
                 "label": id2label[p]}
                for i, p in enumerate(preds)
            ]
            return JSONResponse(content={"result": result})
            
        finally:
            # 清理暫時檔案
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except Exception as e:
        # 添加更詳細的錯誤訊息
        import traceback
        error_details = traceback.format_exc()
        print(f"Error: {str(e)}")
        print(f"Traceback: {error_details}")
        return JSONResponse(status_code=500, content={"error": f"Internal server error: {str(e)}"})