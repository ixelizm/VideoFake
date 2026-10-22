import threading
from typing import Any, Optional, List
import insightface
import numpy

import roop.globals
from roop.typing import Frame, Face

FACE_ANALYSER = None
THREAD_LOCK = threading.Lock()
_USING_CPU_FALLBACK = False


def _provider_names(providers: List[Any]) -> List[str]:
    names = []
    for provider in providers:
        if isinstance(provider, (list, tuple)):
            names.append(provider[0])
        else:
            names.append(provider)
    return names


def get_face_analyser() -> Any:
    global FACE_ANALYSER

    with THREAD_LOCK:
        if FACE_ANALYSER is None:
            providers = roop.globals.execution_providers
            FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=providers)
            FACE_ANALYSER.prepare(ctx_id=0)
    return FACE_ANALYSER


def clear_face_analyser() -> Any:
    global FACE_ANALYSER

    FACE_ANALYSER = None


def _switch_to_cpu_providers() -> None:
    global FACE_ANALYSER, _USING_CPU_FALLBACK
    if _USING_CPU_FALLBACK:
        return
    print('[ROOP.FACE-ANALYSER] CUDA failed — switching onnxruntime providers to CPU.', flush=True)
    roop.globals.execution_providers = ['CPUExecutionProvider']
    FACE_ANALYSER = None
    _USING_CPU_FALLBACK = True
    # Drop any cached CUDA face-swapper so it is rebuilt on CPU too
    try:
        from roop.processors.frame.face_swapper import clear_face_swapper
        clear_face_swapper()
    except Exception:
        pass


def get_one_face(frame: Frame, position: int = 0) -> Optional[Face]:
    many_faces = get_many_faces(frame)
    if many_faces:
        try:
            return many_faces[position]
        except IndexError:
            return many_faces[-1]
    return None


def get_many_faces(frame: Frame) -> Optional[List[Face]]:
    global FACE_ANALYSER
    try:
        return get_face_analyser().get(frame)
    except ValueError:
        return None
    except Exception as exc:
        # CUDA/cuDNN EP failures leave a broken session; rebuild on CPU once.
        msg = str(exc)
        if 'CUDA' in msg or 'CUDNN' in msg or 'EP_FAIL' in msg or 'onnxruntime' in msg.lower():
            with THREAD_LOCK:
                _switch_to_cpu_providers()
                FACE_ANALYSER = insightface.app.FaceAnalysis(
                    name='buffalo_l', providers=roop.globals.execution_providers
                )
                FACE_ANALYSER.prepare(ctx_id=-1)
            try:
                return FACE_ANALYSER.get(frame)
            except Exception:
                return None
        raise


def find_similar_face(frame: Frame, reference_face: Face) -> Optional[Face]:
    many_faces = get_many_faces(frame)
    if many_faces:
        for face in many_faces:
            if hasattr(face, 'normed_embedding') and hasattr(reference_face, 'normed_embedding'):
                distance = numpy.sum(numpy.square(face.normed_embedding - reference_face.normed_embedding))
                if distance < roop.globals.similar_face_distance:
                    return face
    return None
