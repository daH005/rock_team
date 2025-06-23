import tempfile
from http import HTTPStatus

from flask import (
    Flask,
    request,
    render_template,
    abort,
    send_from_directory,
    after_this_request, Response,
)
from werkzeug.utils import secure_filename

from .helpers.stream_save import save_stream
from .recognition_lib.recognition_on_video import recognize_faces_on_video_by_db, RecognitionOnVideoResultType
from .config import (
    HOST,
    PORT,
    DEBUG,
    DB_PATH,
    RESULT_IMAGES_DIR_PATH,
)

app = Flask(__name__)


@app.route('/', methods=['GET'])
def home() -> str:
    return render_template('home.html')


@app.route('/recognize/onVideo', methods=['POST'])
def recognize_on_video() -> RecognitionOnVideoResultType:
    if not request.content_length:
        return abort(HTTPStatus.BAD_REQUEST)

    with tempfile.NamedTemporaryFile(delete=True) as f:
        save_stream(request.stream, f.name)
        result: RecognitionOnVideoResultType = recognize_faces_on_video_by_db(
            f.name,
            DB_PATH,
            RESULT_IMAGES_DIR_PATH,
        )

    return result


@app.route('/takeResultImage/<string:result_image_id>', methods=['GET'])
def take_result_image(result_image_id: str) -> Response:
    result_image_id = secure_filename(result_image_id)

    @after_this_request
    def _remove_file(response: Response) -> Response:
        try:
            RESULT_IMAGES_DIR_PATH.joinpath(result_image_id).unlink()
        except Exception as e:
            app.logger.error(f'Error deleting file: {e}')
        return response

    return send_from_directory(RESULT_IMAGES_DIR_PATH, result_image_id)


if __name__ == '__main__':
    app.run(HOST, PORT, debug=DEBUG)
