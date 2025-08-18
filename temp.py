import os
import time
import math
import pandas as pd
import opensim as osim
from collections import OrderedDict
import psutil, os

p = psutil.Process()                    # 目前這個 Python 程序
print("before:", p.cpu_affinity())      # 讀取目前綁定的 CPU index

p.nice(psutil.REALTIME_PRIORITY_CLASS)  
# import pdb;pdb.set_trace()
# ==== 使用者可調 ====
XML_PATH   = r"C:\Users\MyUser\Desktop\NTKCAP\Patient_data\tempreal\opensim\IK_Setup_Pose2Sim_Halpe26.xml"
XLSX_PATH  = r"C:\Users\MyUser\Desktop\NTKCAP\Patient_data\tempreal\opensim\trcexcel.xlsx"
FPS        = 30.0         # 你的取樣率；原本用 0.033s，所以這裡設 30
EXCEL_IN_MM = False       # 若 Excel 是毫米，改 True（自動轉公尺）

# XML 與資料中 marker 的別名對齊（有需要就加）
ALIAS = {
    "CHip": "Hip",
    # "SomeNameInXML": "NameInData",
}

# 沒有表頭的 Excel：一列為一幀，依序 3 欄為 (x,y,z)
MARKER_NAMES = [
    "Hip", "RHip", "RKnee", "RAnkle", "RBigToe", "RSmallToe", "RHeel",
    "LHip", "LKnee", "LAnkle", "LBigToe", "LSmallToe", "LHeel",
    "Neck", "Head", "Nose", "RShoulder", "RElbow", "RWrist",
    "LShoulder", "LElbow", "LWrist"
]

def _get_tool_attr(obj, *names):
    for n in names:
        if hasattr(obj, n):
            val = getattr(obj, n)
            return val() if callable(val) else val
    raise AttributeError(f"None of {names} found on {type(obj).__name__}")

def _resolve_relative(base_dir, path_in_xml):
    if os.path.isabs(path_in_xml):
        return path_in_xml
    return os.path.abspath(os.path.join(base_dir, path_in_xml))

def build_marker_dict_from_row(row_vals, marker_names, scale=1.0):
    """row_vals: 一列的純數字（無表頭），每 3 欄 = x,y,z"""
    md = OrderedDict()
    for i, name in enumerate(marker_names):
        x = float(row_vals[3*i + 0]) * scale
        y = float(row_vals[3*i + 1]) * scale
        z = float(row_vals[3*i + 2]) * scale
        md[name] = (x, y, z)
    return md
def _coord_is_locked(coord, state):
    # 先試新版：依 state 查
    try:
        return bool(coord.getLocked(state))
    except Exception:
        pass
    # 舊版：無參數屬性
    try:
        return bool(coord.get_locked())
    except TypeError:
        # 少數版本 get_locked(int) 原型
        try:
            return bool(coord.get_locked(0))
        except Exception:
            return False
    except Exception:
        return False

def _coord_lock(coord, state, val=True):
    # 新版：鎖在 state 上
    try:
        coord.setLocked(state, val)
        return
    except Exception:
        pass
    # 舊版：屬性 setter
    try:
        coord.set_locked(val)
    except Exception:
        pass
def main():
    xml_path = os.path.abspath(XML_PATH)
    xml_dir = os.path.dirname(xml_path)

    # IKTool 與 model
    iktool = osim.InverseKinematicsTool(xml_path)
    model_path = _resolve_relative(xml_dir, _get_tool_attr(iktool, "getModelFileName", "get_model_file"))
    model = osim.Model(model_path)
    state = model.initSystem()

    # 讀 Excel（無表頭）
    df = pd.read_excel(XLSX_PATH, sheet_name=0, header=None)
    nframes = len(df)
    scale = 0.001 if EXCEL_IN_MM else 1.0

    # 輸出 .mot
    coordset   = model.getCoordinateSet()
    out_labels = [coordset.get(i).getName() for i in range(coordset.getSize())]
    out_tbl    = osim.TimeSeriesTable()
    out_tbl.setColumnLabels(out_labels)

    # 固定資料
    tasks = iktool.getIKTaskSet()
    dt = 1.0 / FPS
    # nframes = 1000
    # lll = 0
    # while True:
    for lll in range(nframes):
        lll = lll+1
        llll = lll%223
        a = time.time()
        # ---- 1) 由 Excel 取本幀 marker_dict ----
        row_vals = list(df.iloc[llll].values)
        marker_dict = build_marker_dict_from_row(row_vals, MARKER_NAMES, scale=scale)
        labels = list(marker_dict.keys())
        label_set = set(labels)
        t_frame = lll * dt

        # ---- 2) 單幀 TimeSeriesTableVec3 ----
        tbl = osim.TimeSeriesTableVec3()
        tbl.setColumnLabels(labels)
        tbl.appendRow(float(t_frame), osim.RowVectorVec3(len(labels)))
        row_tbl = tbl.updRowAtIndex(0)
        for j, name in enumerate(labels):
            x, y, z = marker_dict[name]
            try:
                row_tbl.set(j, osim.Vec3(x, y, z))
            except Exception:
                row_tbl[j] = osim.Vec3(x, y, z)
        full_table = tbl

        # ---- 3) Marker weights（只加本幀真的有的，含別名對齊）----
        mweights = osim.SetMarkerWeights()
        for k in range(tasks.getSize()):
            mt = osim.IKMarkerTask.safeDownCast(tasks.get(k))
            if mt is None: 
                continue
            if not _get_tool_attr(mt, "getApply", "getOn"):
                continue
            name_xml = _get_tool_attr(mt, "getName")
            name_use = ALIAS.get(name_xml, name_xml)
            if name_use in label_set:
                w = float(_get_tool_attr(mt, "getWeight"))
                mweights.cloneAndAppend(osim.MarkerWeight(name_use, w))
        # 沒有抓到任何權重時：給本幀每個 marker 預設 1.0
        if mweights.getSize() == 0:
            for n in labels:
                mweights.cloneAndAppend(osim.MarkerWeight(n, 1.0))

        # ---- 4) Coordinate refs（依 XML 設定）----
        coordRefs = osim.SimTKArrayCoordinateReference()
        for k in range(tasks.getSize()):
            ct = osim.IKCoordinateTask.safeDownCast(tasks.get(k))
            if ct is None:
                continue
            if not _get_tool_attr(ct, "getApply", "getOn"):
                continue
            try:
                cname = _get_tool_attr(ct, "getCoordinateName", "getName")
            except Exception:
                cname = ct.getName()
            vtype = "default_value"
            if hasattr(ct, "getValueType") or hasattr(ct, "getValueTypeAsString"):
                try:
                    vtype = str(_get_tool_attr(ct, "getValueTypeAsString", "getValueType"))
                except Exception:
                    vtype = "default_value"
            if "manual" in vtype:
                value = float(_get_tool_attr(ct, "getValue"))
            elif "default" in vtype or vtype == "0":
                value = float(coordset.get(cname).getDefaultValue())
            else:
                continue
            weight = float(_get_tool_attr(ct, "getWeight", "get_weight")) if (hasattr(ct, "getWeight") or hasattr(ct, "get_weight")) else 0.0
            cref = osim.CoordinateReference(cname, osim.Constant(value))
            try:    cref.setWeight(weight)
            except: cref.set_weight(weight)
            coordRefs.push_back(cref)

        # ---- 5) Solver ----
        mref = osim.MarkersReference(full_table, mweights)
        solver = osim.InverseKinematicsSolver(model, mref, coordRefs)

        # 帶入 XML 的數值
        # try: 
        # solver.setAccuracy(iktool.get_accuracy())
        solver.setAccuracy(0.00001)   
        # except Exception: pass
        # solver.setConstraintWeight(iktool.get_constraint_weight())
        # solver.setConstraintWeight(inf)
        # except Exception: pass
        # import pdb;pdb.set_trace()
        # 鎖回模型原本 locked 的座標
        for i in range(coordset.getSize()):
            c = coordset.get(i)
            if _coord_is_locked(c, state):
                _coord_lock(c, state, True)

        # === 關鍵：把 solver/mref/state 的時間對齊成 t_frame ===
        try:
            solver.setCurrentTime(t_frame)
        except Exception:
            pass
        try:
            mref.setCurrentTime(t_frame)
        except Exception:
            pass
        state.setTime(t_frame)

        # ---- 6) 組裝 + 寫出本幀到 out_tbl ----
        solver.assemble(state)
        model.realizePosition(state)

        row_out = osim.RowVector(len(out_labels))
        for j, cname in enumerate(out_labels):
            c = coordset.get(cname)
            v = c.getValue(state)
            if cname.endswith(("_tx","_ty","_tz")) or cname.startswith("Abs_t"):
                row_out[j] = v                    # m
            else:
                row_out[j] = v * 180.0 / math.pi  # deg
        out_tbl.appendRow(float(t_frame), row_out)
        print(time.time()-a)

    out_path = os.path.join(xml_dir, "ik_result.mot")
    osim.STOFileAdapter.write(out_tbl, out_path)
    print("IK results written to", out_path)

if __name__ == "__main__":
    main()