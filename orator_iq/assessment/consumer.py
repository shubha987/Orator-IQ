import json
from channels.generic.websocket import AsyncWebsocketConsumer

class VideoConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        data = json.loads(text_data)
        video_data = data['video']

        # Save the video data to a file or process it as needed
        with open('recorded_video.webm', 'wb') as f:
            f.write(video_data)

        await self.send(text_data=json.dumps({
            'status': 'success'
        }))