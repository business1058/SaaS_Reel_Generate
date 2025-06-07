# Snippet from: shake_trans.py
# Generated for client preview - Full code available after payment
# Contact for complete implementation

def create_shake_transition_video(
    image1_path: str,
    image2_path: str,
    output_path: str,
    # Full code available after Monday payment confirmation
    # Contact for complete implementation


def apply_shake_transition_to_video(
    image1,
    image2,
    output_path: str,
    # Full code available after Monday payment confirmation
    # Contact for complete implementation


    def clamp(value: float, min_val: float, max_val: float) -> float:
        return min(max(min_val, value), max_val)

    # Full code available after Monday payment confirmation


    def mix(x: float, y: float, a: float) -> float:
        return x + (y - x) * a

    # Full code available after Monday payment confirmation


    def step(edge0: float, edge1: float, value: float) -> float:
        return min(max(0, (value - edge0) / (edge1 - edge0)), 1)

    # Full code available after Monday payment confirmation


    def smoothstep(edge0: float, edge1: float, value: float) -> float:
        t = min(max(0, (value - edge0) / (edge1 - edge0)), 1)
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def sine_in(t: float) -> float:
        return 1 - math.cos(math.pi * t * 0.5)

    # Full code available after Monday payment confirmation


    def sine_out(t: float) -> float:
        return math.sin(math.pi * t * 0.5)

    # Full code available after Monday payment confirmation


    def sine_in_out(t: float) -> float:
        return -(math.cos(math.pi * t) - 1) * 0.5

    # Full code available after Monday payment confirmation


    def quad_in(t: float) -> float:
        return t * t

    # Full code available after Monday payment confirmation


    def quad_out(t: float) -> float:
        return (2 - t) * t

    # Full code available after Monday payment confirmation


    def quad_in_out(t: float) -> float:
        return 2 * t * t if t < 0.5 else t * (4 - t - t) - 1

    # Full code available after Monday payment confirmation


    def cubic_in(t: float) -> float:
        return t * t * t

    # Full code available after Monday payment confirmation


    def cubic_out(t: float) -> float:
        t = 1 - t
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def cubic_in_out(t: float) -> float:
        if t < 0.5:
            return 4 * t * t * t
        else:
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def __init__(self):
        self._tracks = {}
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def add_frames(self, layer_name: str, layer_data: Dict[str, Any]):
        layer_data = layer_data["layer0"]
        self._add_frames_vec2(layer_name, layer_data, "position")
        self._add_frames_vec2(layer_name, layer_data, "anchor")
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def get(self, path: str, frame: float, normalized: bool = False) -> Optional[float]:
        track = self._tracks.get(path)
        if not track:
            return None

        if not hasattr(track, "keyframe"):
            frame = Utils.clamp(frame, 0, len(track) - 1)
            f0 = int(math.floor(frame))
            f1 = int(math.ceil(frame))
            # Full code available after Monday payment confirmation
            # Contact for complete implementation


    def _add_frames_vec2(self, name: str, data: Dict[str, Any], attr: str):
        attr_data = data.get(attr)
        if not attr_data:
            return

        path = f"{name}.{attr}"

        if isinstance(attr_data, list):
            self._tracks[path] = attr_data
            # Full code available after Monday payment confirmation
            # Contact for complete implementation


    def _interpolate_frame(f0: float, f1: float, t: float) -> float:
        if isinstance(f0, (int, float)) and isinstance(f1, (int, float)):
            return Utils.mix(f0, f1, t)

        if isinstance(f0, list) and isinstance(f1, list):
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def __init__(self, width: int = 1080, height: int = 1920):
        self.width = width
        self.height = height

        self.ctx = moderngl.create_context(standalone=True)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        self.FRAMES = 19
        self.DESIGN_W = 800
        self.DESIGN_H = 800
        self.duration = 1.1
        self.frame_rate = 25

        self.src0_data = {
            "compDuration": 0.8,
            "frameRate": 25,
            "layer0": {
                "frameCount": 20,
                "position": {
                    "x": [
                        363.213191422601,
                        363.153823618598,
                        362.67914316804,
                        361.196101397365,
                        357.956302832473,
                        352.39863125013,
                        345.78041109725,
                        348.979328604241,
                        387.89694900929,
                        453.590738379377,
                        446.802983654209,
                        355.411005029517,
                        265.312092049525,
                        306.105087954409,
                        349.236575725952,
                        373.963191422601,
                        360.657929077528,
                        356.577941185742,
                        360.775277963487,
                        363.713191422601,
                    ],
                    "y": [
                        376.080946639853,
                        376.159318415763,
                        376.791445819725,
                        378.83319989871,
                        383.710073887443,
                        393.687950790519,
                        413.100662076998,
                        446.986974584789,
                        479.816084250392,
                        447.471778729371,
                        361.1045878044,
                        333.281297001702,
                        338.411530475457,
                        370.1083801249,
                        389.862271031809,
                        388.580946639853,
                        381.129199388151,
                        373.543447955266,
                        375.780372568,
                        376.580946639853,
                    ],
                },
                "anchor": {"x": [0.624375] * 20, "y": [0.8325] * 20},
                "scale": {"x": [1] * 20, "y": [1] * 20},
                "rotate": [0] * 20,
                "opacity": [1] * 20,
            },
        }

        self.src1_data = {
            "compDuration": 0.8,
            "frameRate": 25,
            "layer0": {
                "frameCount": 20,
                "position": {
                    "x": [
                        363.213191422601,
                        363.153823618598,
                        362.67914316804,
                        361.196101397365,
                        357.956302832473,
                        352.39863125013,
                        345.78041109725,
                        348.979328604241,
                        387.89694900929,
                        453.590738379377,
                        446.802983654209,
                        355.411005029517,
                        # Full code available after Monday payment confirmation
                        # Contact for complete implementation


    def _load_image_texture(self, image_input) -> moderngl.Texture:
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Could not load image: {image_input}")
        elif isinstance(image_input, Image.Image):
            image = np.array(image_input)
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, np.ndarray):
            image = image_input
        else:
            raise ValueError("Image input must be file path, PIL Image, or numpy array")

    # Full code available after Monday payment confirmation
    # Contact for complete implementation


    def _create_shaders(self):
        motion_vert = """
        #version 330 core
        
        uniform vec2 u_screen_size;
        uniform vec2 u_position;
        uniform vec2 u_anchor;
        
        in vec2 attPosition;
        in vec2 attUV;
        
        out vec2 v_uv;
        
        vec2 transform(vec2 screen_size, vec2 image_size, vec2 translate, vec2 anchor, vec2 scale, float rotate, vec2 uv) {
            float R = rotate * 0.01745329251;
            float c = cos(R);
            float s = sin(R);
            
            vec2 rx = vec2(c, s);
            vec2 ry = vec2(-s, c);
            
            vec2 origin = translate * screen_size;
            vec2 p = uv * screen_size - origin;
            p = vec2(dot(rx, p), dot(ry, p));
            p /= image_size * scale;
            p += anchor;
            return p;
        }
        
        void main() {
            v_uv = transform(u_screen_size, u_screen_size, u_position, u_anchor, vec2(1.0), 0.0, attUV);
            gl_Position = vec4(attPosition, 0.0, 1.0);
        }
        """

        motion_frag = """
        #version 330 core
        
        uniform sampler2D u_src0;
        uniform sampler2D u_src1;
        uniform float u_select;
        
        in vec2 v_uv;
        out vec4 fragColor;
        
        vec4 texture2Dmirror(sampler2D tex, vec2 uv) {
            uv = mod(uv, 2.0);
            uv = mix(uv, 2.0 - uv, step(vec2(1.0), uv));
            return texture(tex, fract(uv));
        }
        
        void main() {
            vec4 src0 = texture2Dmirror(u_src0, v_uv);
            vec4 src1 = texture2Dmirror(u_src1, v_uv);
            fragColor = mix(src0, src1, u_select);
        }
        """

        blur_vert = """
        #version 330 core
        
        uniform float u_step_x;
        uniform float u_step_y;
        uniform float u_direction;
        uniform float u_intensity;
        
        in vec2 attPosition;
        in vec2 attUV;
        
        out vec2 v_uv[9];
        
        #define PI 3.14159265359
        
        void main() {
            float a = PI * u_direction;
            float s = sin(a);
            float c = cos(a);
            
    # Full code available after Monday payment confirmation
    # Contact for complete implementation


    def _create_buffers(self):
        vertices = np.array(
            [
                -1.0,
                -1.0,
                0.0,
                0.0,
                1.0,
                -1.0,
                1.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                -1.0,
                -1.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                -1.0,
                1.0,
                # Full code available after Monday payment confirmation
                # Contact for complete implementation


    def calculate_shake_parameters(self, t: float) -> Dict[str, Any]:
        f = t * self.FRAMES

        if f < 8:
            x = self.ae.get("src0.position.x", f) / self.DESIGN_W
            y = self.ae.get("src0.position.y", f) / self.DESIGN_H
            select_value = 0.0
        else:
            x = self.ae.get("src1.position.x", f) / self.DESIGN_W
            y = self.ae.get("src1.position.y", f) / self.DESIGN_H
            select_value = 1.0

        s = min(self.width, self.height) / 800
        step_x = s / self.width
        step_y = s / self.height

        r = self.ae.get("src1.scale.x", f) or 0
        r = math.radians(r)
        a = self.width / self.height
        dir_x = math.cos(r) / a
        dir_y = math.sin(r)
        r = math.atan2(dir_y, dir_x)
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def render_frame(
        self, texture1: moderngl.Texture, texture2: moderngl.Texture, t: float
    # Full code available after Monday payment confirmation


    def cleanup(self):
        for fb in self.framebuffers:
            fb.release()
        for tex in self.textures:
            tex.release()
        self.vbo.release()
        self.vao.release()
        # Full code available after Monday payment confirmation
        # Contact for complete implementation

