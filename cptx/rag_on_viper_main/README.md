# RAG on VIPER Main

這是一個基於 Streamlit 的檢索增強生成系統，運行在 Neuchips Viper 上，整合了多種大型語言模型和向量搜尋功能。

## 🚀 功能特色

### 核心功能
- **多模型支援**：支援 Qwen3-8B、Qwen2.5-7B、DeepSeek-R1-Distill-Llama-8B 等模型
- **RAG 系統**：完整的檢索增強生成流程，包含文檔解析、向量化和檢索
- **多語言支援**：支援繁體中文、簡體中文和英文介面

### RAG 數據管理
- **文檔上傳**：支援 PDF 文件批量上傳和處理
- **元素提取**：自動提取文檔中的文字和表格元素
- **版本控制**：RAG 數據庫版本管理和切換
- **數據匯出**：支援 JSON 和 PyTorch 格式的數據匯出

## 🛠️ 安裝與設置

### 環境要求
- **作業系統**：Linux (推薦) 或 Windows
- **Python**：3.8+
- **conda/miniconda**：用於環境管理

### 快速安裝

1. **設定主機環境**
   
   確保 NeuChips SDK 已經安裝完成，並且建立好 `neullamacpp` 的 conda 環境 (參考 [Neuchips SDK 安裝指南](https://github.com/neuchips-support/neuchips-sdk/tree/main/sdk_linux/script/neullamacpp))。
   
   系統需要額外安裝以下套件：
   ```bash
   sudo apt-get install tesseract-ocr poppler-utils portaudio19-dev alsa-base alsa-utils pulseaudio
   ```
   
   如果要分析不同語系的 PDF 文檔，需要另外安裝 tesseract 的語言套件，例如簡體中文支援：
   ```bash
   sudo apt install tesseract-ocr-chi-sim
   ```

2. **更新 conda 環境**
   
   使用專案提供的環境配置文件來更新現有的 `neullamacpp` 環境：
   ```bash
   conda env update --name neullamacpp --file ../conda/app_env.yml
   ```

3. **配置數據路徑**
   
   根據您的系統修改 `llm_rag_config.py` 中的路徑設定：
   ```python
   def get_data_path():
       if platform.system() == "Linux":
           return f"/data/"
       elif platform.system() == "Windows":
           return f"C:/data/" or f"D:/data/"
   ```

### 模型準備
在指定的數據路徑下準備 GGUF 格式的模型文件：
```
/data/gguf/neuchips/
├── qwen/
│   ├── Qwen3-8B-F16.gguf
│   └── Qwen2.5-7B-Instruct-F16.gguf
└── DeepSeek-R1-Distill-Llama-8B/
    └── DeepSeek-R1-Distill-Llama-8B-F16.gguf
```

> **注意**：請確保模型文件路徑與 `llm_rag_config.py` 中的設定一致。

## 🚀 使用方法

### 啟動應用程式

**Linux/macOS:**
```bash
# 本地運行（預設簡體中文）
./run_streamlit_local.sh

# 指定語言運行
./run_streamlit_local.sh zh-TW  # 繁體中文
./run_streamlit_local.sh en     # 英文

# 遠程運行（允許外部訪問）
./run_streamlit_remote.sh zh-CN
```

**Windows:**
```cmd
run_streamlit_local.bat
```

> **提示**：首次啟動可能需要較長時間來載入模型和初始化環境。

### 主要使用流程

1. **模型選擇**
   - 在側邊欄選擇所需的 LLM 模型
   - 選擇推理引擎（VIPER 或 CPU）

2. **RAG 功能設置**
   - 開啟 RAG 功能開關
   - 管理 RAG 數據：上傳 PDF 文件
   - 選擇要使用的 RAG 數據版本
   - 提交配置以準備向量數據庫

3. **開始對話**
   - 在聊天界面輸入問題
   - 系統會自動檢索相關文檔並生成回答
   - 支援語音輸入功能

### RAG 數據管理

1. **上傳文件**
   - 點擊 "Manage the Rag Data" 按鈕進入管理頁面
   - 選擇 "Upload File" 標籤
   - 批量選擇並上傳 PDF 文件
   - 系統會自動解析文檔並提取文字與表格元素

2. **創建 RAG 版本**
   - 切換至 "Rag Data Management" 標籤
   - 從已上傳的文件中選擇要包含的檔案
   - 輸入有意義的版本名稱
   - 點擊 "Create Rag Data" 完成建立

3. **管理現有數據**
   - 瀏覽所有已建立的 RAG 數據版本
   - 下載數據（支援 JSON 或 .pt 格式）
   - 刪除不再需要的版本以節省空間

## ⚙️ 配置選項

### LLM 配置 (`llm_rag_config.py`)
```python
llm_config = {
    "model_name": {
        "path": "模型文件路徑",
        "precompiler_cache_path": "預編譯緩存路徑",
        "chat_format": "聊天格式",
        "prompt_template": "提示詞模板",
    }
}
```

### RAG 配置
```python
# RAG 數據庫路徑
rag_db_path = "./data/db/"

# 上傳文件保存路徑
save_path = "./data/uploaded_files/"

# 用戶活動記錄
user_activity_logging = False
```

## 🎨 UI 主題

應用程式使用深色主題，配置文件位於 `.streamlit/config.toml`：
```toml
[theme]
base = "dark"
primaryColor = "#AADC32"
```

## 📊 性能監控

- **TPS 計算**：實時顯示每秒 token 生成數
- **推理時間**：顯示模型推理耗時
- **用戶活動記錄**：可選的操作日誌記錄

## 🐛 故障排除

### 常見問題

1. **模型加載失敗**
   - 檢查模型文件路徑是否正確
   - 確認 GGUF 文件完整性

2. **VIPER 設備未識別**
   - 確認 llama-cpp-python 版本支援 neu_vss_c
   - 檢查設備驅動程式安裝

3. **RAG 向量搜尋錯誤**
   - 確認 RAG 數據已正確載入
   - 檢查向量維度配置

### 日誌查看
啟用用戶活動記錄後，可在日誌中查看詳細的操作記錄，有助於問題診斷和性能分析。

## 📁 專案結構

```
rag_on_viper_main/
├── neuTorch_main.py           # 主應用程式入口點
├── rag_db_app.py              # RAG 數據庫管理應用
├── rag_db_func.py             # RAG 數據庫操作函數
├── rag_db_operations.py       # RAG 數據庫核心操作
├── llm_rag_config.py          # LLM 和 RAG 配置文件
├── language.py                # 多語言翻譯支援
├── user_logger.py             # 用戶活動記錄
├── tps_calculator.py          # TPS（每秒 token 數）計算器
├── run_streamlit_local.sh     # 本地運行腳本 (Linux)
├── run_streamlit_local.bat    # 本地運行腳本 (Windows)
├── run_streamlit_remote.sh    # 遠程運行腳本
├── .streamlit/                # Streamlit 配置
│   └── config.toml            # UI 主題配置
├── resources/                 # 資源文件
└── tests/                     # 測試文件
```

## 🔗 相關連結

- [Neuchips SDK 文檔](https://github.com/neuchips-support/neuchips-sdk)
- [技術支援](https://github.com/neuchips-support/neuchips-sdk/issues)
