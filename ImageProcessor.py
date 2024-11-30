import base64
import io
import json
import requests
from PIL import Image


class ImageProcessor:
    def __init__(self, api_url, model="llama3.2-vision", headers=None):
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

    def resize_image(self, image):
        """
        Redimensiona la imagen para que ninguna dimensión supere el tamaño máximo permitido.
        - image: Objeto PIL.Image.
        - Retorna: Objeto PIL.Image redimensionado.
        """
        if image.size[0] > self.max_pixels[0] or image.size[1] > self.max_pixels[1]:
            image.thumbnail(self.max_pixels, Image.Resampling.LANCZOS)

        return image

    def encode_image(self, image_path):
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

    def send_request(self, prompt, image_path, stream=False):
        """
        Envía una solicitud a la API con el prompt y la imagen proporcionados.
        - prompt: Texto que describe la tarea a realizar.
        - image_path: Ruta al archivo de la imagen.
        - stream: Bandera para habilitar transmisión en tiempo real (por defecto False).
        - Retorna: Respuesta procesada de la API.
        """
        # Codificar la imagen en Base64
        encoded_image = self.encode_image(image_path)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "format": "json",
            "images": [encoded_image],
        }

        response = requests.post(self.api_url, data=json.dumps(payload), headers=self.headers)

        return self.handle_response(response)

    @staticmethod
    def handle_response(response):
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


def process_cats(api_processor, image_path):
    """
    Procesa una imagen de gatos para contar cuántos hay.
    - api_processor: Instancia de la clase ImageProcessor.
    - image_path: Ruta al archivo de la imagen.
    """
    prompt = "Count the number of cats. Output only the number as a int. Be fast and precise please."
    response = api_processor.send_request(prompt, image_path)
    print("Respuesta de API para gatos:", response)


def process_ticket(api_processor, image_path):
    """
    Procesa un ticket para extraer datos estructurados en formato JSON.
    - api_processor: Instancia de la clase ImageProcessor.
    - image_path: Ruta al archivo de la imagen.
    """
    prompt = (
        "Lee el ticket y extrae los datos en formato JSON. "
        "Para cada fila del ticket, devuelve un objeto con los campos: "
        "- 'quantity': cantidad comprada (un número entero, a la izquierda del nombre del producto. No confundir con el indice de productos), "
        "- 'name': nombre del producto (una cadena de texto limpia sin abreviaturas innecesarias), "
        "- 'price_unit': precio por unidad (un número flotante), "
        "Devuelve el JSON como un array de objetos, sin texto adicional."
    )
    response = api_processor.send_request(prompt, image_path)

    if isinstance(response, dict):
        print("Respuesta de API para ticket:", json.dumps(response, indent=4, ensure_ascii=False))
    else:
        print("Respuesta de API para ticket:", response)


if __name__ == "__main__":
    # URL del endpoint de la API
    api_url = "http://ollama.server.local:11434/api/generate"

    processor = ImageProcessor(api_url)

    # Procesar imágenes
    process_cats(processor, "images/cat_5.jpg")

    # Procesar un ticket para extraer datos
    process_ticket(processor, "images/ticket.jpg")
