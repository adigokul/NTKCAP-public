# NTKCAP 安裝腳本使用說明

## 基本用法

### 1. 默認安裝 (使用 Poetry)
```powershell
.\setup.ps1
```

### 2. 使用直接 pip 安裝 (不使用 Poetry)
```powershell
.\setup.ps1 -UseDirectPip
```

### 3. 自定義環境名稱
```powershell
.\setup.ps1 -CondaEnvName "my_custom_env"
```

### 4. 跳過 CUDA 檢查
```powershell
.\setup.ps1 -SkipCudaCheck
```

### 5. 跳過 Poetry 依賴安裝
```powershell
.\setup.ps1 -SkipPoetry
```

### 6. 強制重新創建環境
```powershell
.\setup.ps1 -ForceRecreateEnv
```

### 7. 跳過 TensorRT 模型部署
```powershell
.\setup.ps1 -SkipTensorRTDeploy
```

### 8. 自動化安裝（跳過所有互動提示）
```powershell
.\setup.ps1 -AutoYes
```

### 9. 完全自動化的快速安裝
```powershell
.\setup.ps1 -CondaEnvName "ntkcap_fast" -UseDirectPip -AutoYes
```

## 組合參數

### 使用自定義環境名稱 + 直接 pip 安裝
```powershell
.\setup.ps1 -CondaEnvName "ntkcap_custom" -UseDirectPip
```

### 跳過 CUDA 檢查 + 使用直接 pip 安裝
```powershell
.\setup.ps1 -SkipCudaCheck -UseDirectPip
```

## 參數說明

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `CondaEnvName` | string | "ntkcap_env" | 設定 conda 環境名稱 |
| `UseDirectPip` | switch | false | 使用直接 pip 安裝而不是 Poetry |
| `SkipCudaCheck` | switch | false | 跳過 CUDA 安裝檢查 |
| `SkipPoetry` | switch | false | 跳過 Poetry 依賴安裝 |
| `ForceRecreateEnv` | switch | false | 強制重新創建現有環境 |
| `SkipTensorRTDeploy` | switch | false | 跳過 TensorRT 模型部署 |
| `AutoYes` | switch | false | 自動回答所有互動提示為 yes |

## 安裝方法比較

### Poetry 安裝 (預設)
- ✅ 統一依賴管理
- ✅ 版本鎖定
- ❌ 安裝時間較長
- ❌ 需要 pyproject.toml

### 直接 pip 安裝 (`-UseDirectPip`)
- ✅ 安裝速度快
- ✅ 順序明確
- ✅ 不需要 Poetry
- ❌ 手動版本管理

## 直接 pip 安裝包順序

當使用 `-UseDirectPip` 時，將按照以下順序安裝：

1. OpenSim (conda)
2. 基礎套件 (bs4, multiprocess, keyboard, import_ipynb, kivy)
3. 核心套件 (Pose2Sim==0.4, numpy==1.21.6, scipy==1.13.0)
4. 應用套件 (ultralytics, tkfilebrowser, matplotlib==3.8.4, pyserial)
5. 工具套件 (func_timeout, pygltflib, natsort, openpyxl, pyqtgraph)
6. PyQt6 套件 (PyQt6==6.7.0, PyQt6-WebEngine==6.7.0)
7. CUDA 套件 (cupy-cuda11x)
8. 最終 numpy 版本 (numpy==1.22.4)

## 激活環境

安裝完成後，使用以下命令激活環境：

```powershell
# 如果使用預設環境名稱
.\activate_ntkcap.ps1

# 如果使用自定義環境名稱
conda activate your_custom_env_name
```

## 互動式提示

腳本中可能會出現以下互動式提示：

1. **CUDA 檢查失敗**：詢問是否在沒有 CUDA 的情況下繼續
2. **環境已存在**：詢問是否重新創建現有環境
3. **無效環境資料夾**：詢問是否移除並重新創建
4. **TensorRT 模型部署**：詢問是否進行模型部署（耗時較長）

### 避免互動式提示

使用 `-AutoYes` 參數可以自動回答所有提示為 "yes"：

```powershell
# 完全自動化安裝，無需任何手動干預
.\setup.ps1 -CondaEnvName "ntkcap_fast" -UseDirectPip -AutoYes
```

或者使用特定參數來控制個別行為：

```powershell
# 強制重新創建環境 + 跳過 TensorRT 部署
.\setup.ps1 -UseDirectPip -ForceRecreateEnv -SkipTensorRTDeploy
```