# Snippet from: glitch_flash.py
# Generated for client preview - Full code available after payment
# Contact for complete implementation

def main():
    app = QApplication(sys.argv)

    # Set OpenGL format
    format = QSurfaceFormat()
    format.setVersion(3, 3)
    format.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    format.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
    # Full code available after Monday payment confirmation
    # Contact for complete implementation


    def __init__(self):
        super().__init__()
        self.setMinimumSize(800, 600)

        # Effect parameters
        self.glitch_intensity = 0.5
        self.time_offset = 0.0
        self.color_separation = 0.02
        self.noise_amount = 0.3
        self.scan_lines = True
        self.digital_noise = True

        # OpenGL objects
        self.shader_program = None
        self.vao = None
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def initializeGL(self):
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.0, 0.0, 0.0, 1.0)

        # Vertex shader
        vertex_shader = """
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec2 aTexCoord;
        
        out vec2 TexCoord;
        
        void main()
        {
            gl_Position = vec4(aPos, 1.0);
            TexCoord = aTexCoord;
        }
        """

        # Fragment shader with glitch effects
        fragment_shader = """
        #version 330 core
        out vec4 FragColor;
        
        in vec2 TexCoord;
        
        uniform sampler2D mainTexture;
        uniform sampler2D noiseTexture;
        uniform float time;
        uniform float glitchIntensity;
        uniform float colorSeparation;
        uniform float noiseAmount;
        uniform bool scanLines;
        uniform bool digitalNoise;
        uniform vec2 resolution;
        
        // Random function
        float random(vec2 st) {
            return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123);
        }
        
        // Noise function
        float noise(vec2 st) {
            vec2 i = floor(st);
            vec2 f = fract(st);
            float a = random(i);
            float b = random(i + vec2(1.0, 0.0));
            float c = random(i + vec2(0.0, 1.0));
            float d = random(i + vec2(1.0, 1.0));
            vec2 u = f * f * (3.0 - 2.0 * f);
            return mix(a, b, u.x) + (c - a)* u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
        }
        
        void main()
        {
            vec2 uv = TexCoord;
            
            // Time-based distortion
            float timeShift = time * 0.5;
            
            // Horizontal glitch displacement
            float glitchLine = sin(uv.y * 800.0 + timeShift * 10.0) * 0.5 + 0.5;
            float glitchNoise = noise(vec2(uv.y * 80.0, timeShift)) * 2.0 - 1.0;
            float displacement = glitchNoise * glitchIntensity * 0.05 * glitchLine;
            
            // Digital block displacement
            float blockY = floor(uv.y * 80.0) / 80.0;
            float blockNoise = noise(vec2(blockY * 100.0, floor(timeShift * 5.0))) * 2.0 - 1.0;
            if (abs(blockNoise) > 0.7) {
                displacement += blockNoise * glitchIntensity * 0.1;
            }
            
            // Apply displacement
            vec2 distortedUV = uv + vec2(displacement, 0.0);
            
            // RGB channel separation (chromatic aberration)
            vec2 redOffset = vec2(colorSeparation * glitchIntensity, 0.0);
            vec2 blueOffset = vec2(-colorSeparation * glitchIntensity, 0.0);
            
            float r = texture(mainTexture, distortedUV + redOffset).r;
            float g = texture(mainTexture, distortedUV).g;
            float b = texture(mainTexture, distortedUV + blueOffset).b;
            float a = texture(mainTexture, distortedUV).a;
            
            vec4 color = vec4(r, g, b, a);
            
            // Scan lines
            if (scanLines) {
                float scanline = sin(uv.y * resolution.y * 2.0) * 0.5 + 0.5;
                scanline = mix(0.8, 1.0, scanline);
                color.rgb *= scanline;
            }
            
            // Digital noise
            if (digitalNoise) {
                float digitalNoiseValue = random(uv + timeShift) * 2.0 - 1.0;
                if (abs(digitalNoiseValue) > 0.9) {
                    color.rgb += digitalNoiseValue * noiseAmount * glitchIntensity;
                }
                
                // Color quantization for digital effect
                color.rgb = floor(color.rgb * 16.0) / 16.0;
            }
            # Full code available after Monday payment confirmation
            # Contact for complete implementation


    def create_default_texture(self):
        # Create a simple gradient texture if no image is loaded
        width, height = 512, 512
        data = np.zeros((height, width, 3), dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                # Create a colorful gradient pattern
                # Full code available after Monday payment confirmation
                # Contact for complete implementation


    def create_noise_texture(self):
        # Create noise texture for static effect
        width, height = 256, 256
        noise_data = np.random.randint(0, 256, (height, width), dtype=np.uint8)
        noise_bytes = noise_data.tobytes()  # Convert to bytes

        self.noise_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.noise_texture)

        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RED,
            # Full code available after Monday payment confirmation
            # Contact for complete implementation


    def update_quad_vertices(self):
        """Update quad vertices to maintain aspect ratio"""
        if self.image_width == 0 or self.image_height == 0:
            return

        # Calculate aspect ratios
        image_aspect = self.image_width / self.image_height
        widget_aspect = self.width() / self.height()

        # Calculate scale factors to fit image while maintaining aspect ratio
        if image_aspect > widget_aspect:
            # Image is wider than widget - fit to width
            scale_x = 1.0
            scale_y = widget_aspect / image_aspect
        else:
            # Image is taller than widget - fit to height
            scale_x = image_aspect / widget_aspect
            scale_y = 1.0

        # Create new vertices with proper scaling
        vertices = np.array(
            [
                # positions                    # texture coords
                -scale_x,
                -scale_y,
                0.0,
                0.0,
                # Full code available after Monday payment confirmation
                # Contact for complete implementation


    def load_texture_data(self, data, width, height):
        try:
            # Ensure OpenGL context is current
            self.makeCurrent()

            # Convert data to bytes if it's a numpy array
            if isinstance(data, np.ndarray):
                # Ensure data is uint8 and contiguous
                data = data.astype(np.uint8)
                if not data.flags["C_CONTIGUOUS"]:
                    data = np.ascontiguousarray(data)
                data_bytes = data.tobytes()
            else:
                data_bytes = data

            if self.texture is None:
                self.texture = glGenTextures(1)

            glBindTexture(GL_TEXTURE_2D, self.texture)

            # Use bytes data instead of numpy array directly
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RGB,
                width,
                # Full code available after Monday payment confirmation
                # Contact for complete implementation


    def load_image(self, filepath):
        try:
            # Load image with PIL
            image = Image.open(filepath)

            # Convert to RGB if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Resize if too large (to prevent memory issues)
            max_size = 2048
            if image.width > max_size or image.height > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            # Convert to numpy array
            data = np.array(image, dtype=np.uint8)

    # Full code available after Monday payment confirmation
    # Contact for complete implementation


    def paintGL(self):
        try:
            if self.shader_program is None:
                return

            glClear(GL_COLOR_BUFFER_BIT)

            glUseProgram(self.shader_program)

            # Set uniforms
            glUniform1f(
                glGetUniformLocation(self.shader_program, "time"), self.time_offset
            )
            glUniform1f(
                glGetUniformLocation(self.shader_program, "glitchIntensity"),
                self.glitch_intensity,
            )
            glUniform1f(
                glGetUniformLocation(self.shader_program, "colorSeparation"),
                self.color_separation,
            )
            glUniform1f(
                glGetUniformLocation(self.shader_program, "noiseAmount"),
                self.noise_amount,
            )
            glUniform1i(
                glGetUniformLocation(self.shader_program, "scanLines"), self.scan_lines
            )
            glUniform1i(
                glGetUniformLocation(self.shader_program, "digitalNoise"),
                self.digital_noise,
            )
            # Full code available after Monday payment confirmation
            # Contact for complete implementation


    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)
        # Update quad vertices when window is resized
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def update_animation(self):
        self.time_offset += 0.016  # Increment time
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def set_glitch_intensity(self, value):
        self.glitch_intensity = value / 100.0

    # Full code available after Monday payment confirmation


    def set_color_separation(self, value):
        self.color_separation = value / 1000.0

    # Full code available after Monday payment confirmation


    def set_noise_amount(self, value):
        self.noise_amount = value / 100.0
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def __init__(self):
        super().__init__()
        self.setWindowTitle("CapCut Glitch Effect - PyQt6 + OpenGL")
        self.setGeometry(100, 100, 1200, 800)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Layout
        layout = QHBoxLayout(central_widget)

        # OpenGL widget
        self.glitch_widget = GlitchEffect()
        layout.addWidget(self.glitch_widget, 2)

        # Controls panel
        controls_layout = QVBoxLayout()
        controls_widget = QWidget()
        controls_widget.setLayout(controls_layout)
        controls_widget.setMaximumWidth(300)
        layout.addWidget(controls_widget)

        # Load image button
        load_btn = QPushButton("Load Image")
        load_btn.clicked.connect(self.load_image)
        controls_layout.addWidget(load_btn)

        # Glitch intensity slider
        controls_layout.addWidget(QLabel("Glitch Intensity:"))
        self.intensity_slider = QSlider(Qt.Orientation.Horizontal)
        self.intensity_slider.setRange(0, 100)
        self.intensity_slider.setValue(50)
        self.intensity_slider.valueChanged.connect(
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def load_image(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def toggle_scan_lines(self):
        self.glitch_widget.scan_lines = not self.glitch_widget.scan_lines

    # Full code available after Monday payment confirmation


    def toggle_digital_noise(self):
        self.glitch_widget.digital_noise = not self.glitch_widget.digital_noise
        # Full code available after Monday payment confirmation
        # Contact for complete implementation

