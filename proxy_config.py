import os
import requests


def set_proxy():
    # 要设置的代理信息
    # proxy_info = {
    #     "https_proxy": "http://172.16.1.193:7890",
    #     "http_proxy": "http://172.16.1.193:7890",
    #     "all_proxy": "socks5://172.16.1.193:7890"
    # }

    proxy_info = {
        "https_proxy": "http://172.16.2.68:7890",
        "http_proxy": "http://172.16.2.68:7890",
        "all_proxy": "socks5://172.16.2.68:7891"
    }

    # proxy_info = {
    #     "https_proxy": "http://172.16.1.191:7890",
    #     "http_proxy": "http://172.16.1.191:7890",
    #     "all_proxy": "socks5://172.16.1.191:7891"
    # }

    # 设置环境变量
    for key, value in proxy_info.items():
        os.environ[key] = value

    # 打印设置后的环境变量
    print("Proxy settings:")
    for key in proxy_info.keys():
        print(f"{key}: {os.environ.get(key)}")

    # 使用代理发送HTTP请求
    response = requests.get("http://httpbin.org/ip")
    print("Response from proxy server:")
    print(response.text)
