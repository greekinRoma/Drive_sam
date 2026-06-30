import numpy as np
import onnxruntime as ort
import onnx
import torch
import argparse
import cv2
def preprocess_image(image_np, target_size=1024):
    """
    将原始图像预处理为模型输入格式
    image_np: HWC, BGR or RGB, 范围 0-255
    返回: (1,3,1024,1024) float32, 归一化到 [0,1]（根据训练时的预处理）
    """
    # 缩放到固定尺寸
    h, w = image_np.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image_np, (new_w, new_h))
    # 填充到 target_size x target_size
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    # BGR -> RGB, HWC -> CHW, 归一化到 [0,1]
    padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    padded = padded.astype(np.float32)
    input_tensor = np.transpose(padded, (2, 0, 1))  # CHW
    input_batch = np.expand_dims(input_tensor, axis=0)  # NCHW
    return input_batch

def test_onnx_model(onnx_path, input_shape=(1,3,1024,1024), use_random_input=True, image_path=None):
    # 1. 验证 ONNX 模型结构
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX 模型结构验证通过")

    # 2. 创建推理会话
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)
    
    # 获取输入输出名称
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"输入节点: {input_name}, 形状: {session.get_inputs()[0].shape}")
    print(f"输出节点: {output_name}, 形状: {session.get_outputs()[0].shape}")

    # 3. 准备输入数据
    if use_random_input:
        # 随机输入
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        print("使用随机输入测试")
    else:
        # 使用真实图像（需提供图像路径）
        if image_path is None:
            raise ValueError("请提供图像路径或使用随机输入")
        img = cv2.imread(image_path)
        cv2.imshow("inp",img)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")
        dummy_input = preprocess_image(img, target_size=input_shape[2])
        print("使用真实图像测试")

    # 4. 执行推理
    try:
        outputs = session.run([output_name], {input_name: dummy_input})
        output = outputs[0]
        print(f"推理成功，输出形状: {output.shape}")
        print(f"输出值范围: [{output.min():.4f}, {output.max():.4f}]")

        mask = (output > 0.).astype(np.int32)*255
        back = np.zeros([3,1024,1024])
        back[:2] = mask
        back=back.transpose(1,2,0)
        cv2.imshow("test",back)
        cv2.waitKey(-1)

        
        
    except Exception as e:
        print(f"推理失败: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_path", type=str, default="segmentation_model.onnx", help="ONNX 文件路径")
    parser.add_argument("--input_shape", type=str, default="1,3,1024,1024", help="输入形状, 如 1,3,1024,1024")
    parser.add_argument("--use_random", action="store_true", help="使用随机输入测试")
    parser.add_argument("--image", default="onnx_export/__347+601.jpg",type=str, help="使用真实图像测试（提供图像路径）")
    args = parser.parse_args()

    shape = tuple(map(int, args.input_shape.split(',')))
    use_random = args.use_random or (args.image is None)
    success = test_onnx_model(args.onnx_path, input_shape=shape, use_random_input=use_random, image_path=args.image)
    exit(0 if success else 1)