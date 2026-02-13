import logging
import datetime
import os

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
log_filename = f"log_{current_time}.log"


log_dir = "./logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    
file_path = os.path.join(log_dir, log_filename)

logging.basicConfig(
    level=logging.INFO,  # 设置日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    filename=file_path,  # 指定生成的文件路径
    filemode='a',        # 模式: 'w' (覆盖) 或 'a' (追加)
    # 设置日志内容格式：时间 - 级别 - 消息
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'  # 设置日志内容里的时间格式
)

logger = logging.getLogger(__name__)