"""Reuse one predictor's image features across prompts and evaluation modes.

Only an identical NumPy buffer/view can hit the cache. The worker owns this buffer,
never mutates it, and explicitly ends the session between images and on errors.
AMG's different crops therefore encode independently and release features normally.
"""
import time


class ImageEmbeddingCache:
    def __init__(self, predictor, sync):
        self.predictor = predictor
        self.sync = sync
        self._image = None
        self._encode_seconds = None
        self.events = []

    def __getattr__(self, name):
        return getattr(self.predictor, name)

    def reset_predictor(self):
        self._image = None
        self._encode_seconds = None
        reset = getattr(self.predictor, 'reset_predictor', None)
        if reset is None:
            reset = self.predictor.reset_image  # SAM 1
        reset()

    reset_image = reset_predictor

    def begin_image(self):
        self.reset_predictor()
        self.events = []

    def set_image(self, image):
        previous = self._image
        same_view = (previous is not None and image.shape == previous.shape
                     and image.strides == previous.strides and image.dtype == previous.dtype
                     and image.__array_interface__['data'][0] == previous.__array_interface__['data'][0])
        if same_view:
            self.events.append({'reused': True, 'seconds': self._encode_seconds})
            return
        self.reset_predictor()
        self.sync()
        start = time.perf_counter()
        self.predictor.set_image(image)
        self.sync()
        self._encode_seconds = time.perf_counter() - start
        self._image = image  # Keep the buffer alive until its features are released.
        self.events.append({'reused': False, 'seconds': self._encode_seconds})

    def timing(self, start_event, actual_seconds):
        events = self.events[start_event:]
        shared = sum(event['seconds'] for event in events if event['reused'])
        return {'inference_seconds': actual_seconds + shared,
                'actual_inference_seconds': actual_seconds,
                'encode_seconds': sum(event['seconds'] for event in events),
                'shared_encode_seconds': shared,
                'encoder_calls': sum(not event['reused'] for event in events),
                'embedding_reuses': sum(event['reused'] for event in events)}
