import base64
import io
import json
import time
from typing import Dict, Optional, Union

import requests
from PIL import Image


class ImageProcessor:
    def __init__(self, api_url: str, model: str = "gemma3:12b", headers: Optional[Dict[str, str]] = None):
        """
        Inicializa el procesador de imágenes.
        - api_url: URL del endpoint de la API.
        - model: Modelo utilizado para el procesamiento (por defecto "llama3.2-vision").
        - headers: Encabezados HTTP para las solicitudes (por defecto incluye 'Content-Type: application/json').
        """
        self.api_url = api_url
        self.model = model
        self.headers = headers or {"Content-Type": "application/json"}
        self.max_pixels = (1120, 1120)

    def resize_image(self, image: Image.Image) -> Image.Image:
        """
        Redimensiona la imagen para que ninguna dimensión supere el tamaño máximo permitido.
        - image: Objeto PIL.Image.
        - Retorna: Objeto PIL.Image redimensionado.
        """
        if image.size[0] > self.max_pixels[0] or image.size[1] > self.max_pixels[1]:
            image.thumbnail(self.max_pixels, Image.Resampling.LANCZOS)

        return image

    def encode_image(self, image_path: str) -> str:
        """
        Codifica una imagen en formato Base64.
        - image_path: Ruta al archivo de la imagen.
        - Retorna: Una cadena codificada en Base64.
        """
        with Image.open(image_path) as img:
            img = self.resize_image(img)  # Redimensionar si es necesario
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format="JPEG", quality=95)
            img_byte_arr.seek(0)

            return base64.b64encode(img_byte_arr.read()).decode("utf-8")

    def send_request(self, prompt: str, image_path: str, stream: bool = False) -> Dict[str, Union[Dict, float]]:
        """
        Envía una solicitud a la API con el prompt y la imagen proporcionados.
        - prompt: Texto que describe la tarea a realizar.
        - image_path: Ruta al archivo de la imagen.
        - stream: Bandera para habilitar transmisión en tiempo real (por defecto False).
        - Retorna: Respuesta procesada de la API.
        """
        encoded_image = self.encode_image(image_path)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "format": "json",
            "images": [encoded_image],
            "verbose": False
        }

        start_time = time.time()
        response = requests.post(self.api_url, data=json.dumps(payload), headers=self.headers)
        end_time = time.time()

        response_time = end_time - start_time

        return {"response": self.handle_response(response), "response_time": response_time}

    @staticmethod
    def handle_response(response: requests.Response) -> Union[Dict, str]:
        """
        Procesa la respuesta de la API.
        - response: Objeto de respuesta HTTP.
        - Retorna: JSON con la respuesta o un mensaje de error.
        """
        if response.status_code == 200:
            try:
                response_json = response.json()

                if isinstance(response_json, str):
                    try:
                        return json.loads(response_json)
                    except json.JSONDecodeError:
                        return {"error": "Respuesta JSON no válida (anidada como texto)"}
                return response_json.get("response", response_json)  # Obtener la clave 'response' si existe
            except json.JSONDecodeError:
                return {"error": "Respuesta JSON no válida"}
        else:
            return {"error": response.status_code, "details": response.text}


def process_cats(api_processor: ImageProcessor, image_path: str) -> None:
    """
    Procesa una imagen de gatos para contar cuántos hay.
    - api_processor: Instancia de la clase ImageProcessor.
    - image_path: Ruta al archivo de la imagen.
    """
    prompt = "计算猫的数量。仅输出整数数字。请快速准确。JSON 输出。"
    result: Dict = api_processor.send_request(prompt, image_path)
    response = result["response"]
    response_time = result["response_time"]

    print("Respuesta de API para gatos:", json.dumps(json.loads(response), indent=4, ensure_ascii=False))
    print(f"Tiempo de respuesta: {response_time:.3f} segundos")


def process_ticket(api_processor: ImageProcessor, image_path: str) -> None:
    """
    Procesa un ticket para extraer datos estructurados en formato JSON.
    - api_processor: Instancia de la clase ImageProcessor.
    - image_path: Ruta al archivo de la imagen.
    """
    prompt = (
        """
        Lee ticket extrae datos formato JSON. 
        productos devuelve objeto 
        quantity ' cantidad comprada número entero izquierda nombre. no indice productos 
        name ' nombre producto cadena texto limpia 
        price _ unit ' precio por unidad número flotante 
        Devuelve JSON array objetos sin texto adicional. 
        """)

    result: Dict = api_processor.send_request(prompt, image_path)
    response = result["response"]
    response_time = result["response_time"]

    calculate_totals((json.loads(response)))
    print(f"Tiempo de respuesta: {response_time:.3f} segundos")


def calculate_totals(json_data: Dict[str, Dict]) -> None:
    """
    Calcula el precio total por producto y el total general.
    - json_data: Datos en formato JSON que contienen productos con precio y cantidad.
    """
    key = list(json_data.keys())[0]
    products = json_data[key]

    total_general = 0

    for product in products:
        product["price_total"] = product["quantity"] * product["price_unit"]
        total_general += product["price_total"]

    print("Detalles con precio total por producto:")
    print(json.dumps(products, indent=4, ensure_ascii=False))
    print(f"Total general: {total_general:.2f}")


if __name__ == "__main__":
    processor = ImageProcessor("http://ollama.lab.server.local/api/generate")
    process_cats(processor, "images/cat_5.jpg")
    process_ticket(processor, "images/ticket.jpg")
