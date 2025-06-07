# Snippet from: glitch (2).py
# Generated for client preview - Full code available after payment
# Contact for complete implementation

def main():
    app = QApplication(sys.argv)
    
    # Set OpenGL format globally
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    QSurfaceFormat.setDefaultFormat(fmt)
    # Full code available after Monday payment confirmation
    # Contact for complete implementation


    def __init__(self):
        super().__init__()
        
        # Set OpenGL format
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        self.setFormat(fmt)
        
        # Glitch parameters
        self.time_factor = 0.0
        self.glitch_intensity = 0.8
        self.rgb_shift_amount = 0.02
        self.scanline_intensity = 0.3
        self.noise_amount = 0.1
        self.block_size = 32.0
        self.distortion_strength = 0.05
        
        # Animation
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def initializeGL(self):
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Vertex shader
        vertex_shader_source = """
        #version 330 core
        layout (location = 0) in vec3 attPosition;
        layout (location = 1) in vec2 attUV;
        
        out vec2 uv0;
        
        void main() {
            gl_Position = vec4(attPosition, 1.0);
            uv0 = attUV;
        }
        """
        
        # Fragment shader - Glitch Level 2 Effect
        fragment_shader_source = """
        #version 330 core
        in vec2 uv0;
        out vec4 FragColor;

        uniform sampler2D inputImageTexture;
        uniform sampler2D noiseTexture;
        uniform float time;
        uniform float glitchIntensity;
        uniform float rgbShiftAmount;
        uniform float scanlineIntensity;
        uniform float noiseAmount;
        uniform float blockSize;
        uniform float distortionStrength;
        uniform vec2 resolution;
        uniform float ghostSeparation;  // NEW: Ghost separation amount (0.0 to 1.0)

        // Random function
        float random(vec2 st) {
            return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
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
            
            return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
        }

        // Digital noise
        float digitalNoise(vec2 uv, float time) {
            float noise1 = noise(uv * 100.0 + time * 2.0);
            float noise2 = noise(uv * 200.0 - time * 1.5);
            return (noise1 + noise2) * 0.5;
        }

        // Edge detection for ghost separation
        float detectEdges(sampler2D tex, vec2 uv) {
            vec2 texelSize = 1.0 / textureSize(tex, 0);
            
            // Sample surrounding pixels
            float tl = length(texture(tex, uv + vec2(-texelSize.x, -texelSize.y)).rgb);
            float tm = length(texture(tex, uv + vec2(0.0, -texelSize.y)).rgb);
            float tr = length(texture(tex, uv + vec2(texelSize.x, -texelSize.y)).rgb);
            float ml = length(texture(tex, uv + vec2(-texelSize.x, 0.0)).rgb);
            float mm = length(texture(tex, uv).rgb);
            float mr = length(texture(tex, uv + vec2(texelSize.x, 0.0)).rgb);
            float bl = length(texture(tex, uv + vec2(-texelSize.x, texelSize.y)).rgb);
            float bm = length(texture(tex, uv + vec2(0.0, texelSize.y)).rgb);
            float br = length(texture(tex, uv + vec2(texelSize.x, texelSize.y)).rgb);
            
            // Sobel operator
            float gx = -tl - 2.0*ml - bl + tr + 2.0*mr + br;
            float gy = -tl - 2.0*tm - tr + bl + 2.0*bm + br;
            
            return sqrt(gx*gx + gy*gy);
        }

        // Create ghost offset based on image content
        vec2 calculateGhostOffset(vec2 uv, float separation) {
            if (separation <= 0.0) return vec2(0.0);
            
            // Use image gradients to determine ghost direction
            vec2 texelSize = 1.0 / textureSize(inputImageTexture, 0);
            
            vec3 left = texture(inputImageTexture, uv - vec2(texelSize.x, 0.0)).rgb;
            vec3 right = texture(inputImageTexture, uv + vec2(texelSize.x, 0.0)).rgb;
            vec3 up = texture(inputImageTexture, uv - vec2(0.0, texelSize.y)).rgb;
            vec3 down = texture(inputImageTexture, uv + vec2(0.0, texelSize.y)).rgb;
            
            // Calculate gradient
            float gradX = length(right - left);
            float gradY = length(down - up);
            
            // Create offset based on content and some randomness
            vec2 contentOffset = vec2(gradX - 0.5, gradY - 0.5) * 2.0;
            vec2 randomOffset = vec2(
                sin(uv.x * 20.0 + time * 3.0),
                cos(uv.y * 15.0 + time * 2.5)
            );
            
            vec2 finalOffset = normalize(contentOffset + randomOffset * 0.3) * separation * 0.03;
            return finalOffset;
        }

        // Glitch blocks
        vec2 glitchBlocks(vec2 uv, float time) {
            vec2 blockUV = floor(uv * blockSize) / blockSize;
            float glitchTime = floor(time * 8.0) / 8.0;
            
            float glitchRandom = random(blockUV + glitchTime);
            
            if (glitchRandom > 0.7) {
                float offsetX = (random(blockUV + glitchTime + 1.0) - 0.5) * distortionStrength;
                float offsetY = (random(blockUV + glitchTime + 2.0) - 0.5) * distortionStrength * 0.1;
                uv.x += offsetX;
                uv.y += offsetY;
            }
            
            return uv;
        }

        // RGB shift
        vec3 rgbShift(sampler2D tex, vec2 uv, float amount) {
            float r = texture(tex, uv + vec2(amount, 0.0)).r;
            float g = texture(tex, uv).g;
            float b = texture(tex, uv - vec2(amount, 0.0)).b;
            return vec3(r, g, b);
        }

        // Scanlines
        float scanlines(vec2 uv, float intensity) {
            float scanline = sin(uv.y * resolution.y * 2.0) * 0.5 + 0.5;
            return 1.0 - (scanline * intensity);
        }

        // Horizontal glitch lines
        vec2 horizontalGlitch(vec2 uv, float time) {
            float glitchLine = floor(uv.y * 100.0) / 100.0;
            float glitchTime = floor(time * 10.0);
            float glitch = random(vec2(glitchLine, glitchTime));
            
            if (glitch > 0.9) {
                float offset = (random(vec2(glitchLine, glitchTime + 1.0)) - 0.5) * 0.1;
                uv.x += offset * glitchIntensity;
            }
            
            return uv;
        }

        // Data moshing effect
        vec2 dataMosh(vec2 uv, float time) {
            vec2 blockUV = floor(uv * 20.0) / 20.0;
            float moshTime = floor(time * 3.0);
            # Full code available after Monday payment confirmation
            # Contact for complete implementation


    def get_ghost_state(self, time):
        """Calculate ghost separation distance and opacity"""
        cycle_time = time % self.ghost_frequency
        if cycle_time < self.ghost_duration:
            progress = cycle_time / self.ghost_duration
            # Create out-and-back movement: 0 -> 1 -> 0
            if progress < 0.5:
                # Ghost moving out (0 to 1)
                separation = (progress * 2.0)
            else:
            # Full code available after Monday payment confirmation
            # Contact for complete implementation


    def load_default_texture(self):
        """Load default texture for demonstration"""
        # Create a colorful test pattern
        data = np.zeros((512, 512, 3), dtype=np.uint8)
        for i in range(512):
            for j in range(512):
                # Create a test pattern with gradients and shapes
                r = int((i / 512.0) * 255)
                g = int((j / 512.0) * 255)
                b = int(((i + j) / 1024.0) * 255)
                
                # Add some geometric patterns
                if (i // 64 + j // 64) % 2 == 0:
                    r = min(255, r + 50)
                
                # Add circular pattern
                # Full code available after Monday payment confirmation
                # Contact for complete implementation


    def create_noise_texture(self):
        """Create a noise texture for glitch effects"""
        noise_data = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        
        self.noise_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.noise_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 256, 256, 0, GL_RGB, GL_UNSIGNED_BYTE, noise_data)
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def load_texture_from_file(self, filepath):
        """Load texture from image file"""
        try:
            if not glIsTexture(self.input_texture):
                print("Failed to create texture")
                return False
            img = Image.open(filepath)
            img = img.convert('RGB')
            img_data = np.array(img).copy()
            height, width = img_data.shape[:2]
            
            if self.input_texture:
                glDeleteTextures(1, [self.input_texture])
                
            self.input_texture = glGenTextures(1)
            # Full code available after Monday payment confirmation
            # Contact for complete implementation


    def update_time(self):
        """Update animation time"""
        self.time_factor = time.time() - self.start_time
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def paintGL(self):
        if not self.context().isValid():
            return
        glClear(GL_COLOR_BUFFER_BIT)
        
        if not self.program or not self.input_texture:
            return
            
        glUseProgram(self.program)
        
        # Bind textures
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.input_texture)
        glUniform1i(glGetUniformLocation(self.program, "inputImageTexture"), 0)
        
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.noise_texture)
        glUniform1i(glGetUniformLocation(self.program, "noiseTexture"), 1)
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)

    # Full code available after Monday payment confirmation


    def set_glitch_intensity(self, value):
        self.glitch_intensity = value / 100.0
        
    # Full code available after Monday payment confirmation


    def set_rgb_shift(self, value):
        self.rgb_shift_amount = value / 1000.0
        
    # Full code available after Monday payment confirmation


    def set_scanlines(self, value):
        self.scanline_intensity = value / 100.0
        
    # Full code available after Monday payment confirmation


    def set_noise(self, value):
        self.noise_amount = value / 100.0
        
    # Full code available after Monday payment confirmation


    def set_block_size(self, value):
        self.block_size = float(value)
        
    # Full code available after Monday payment confirmation


    def set_distortion(self, value):
        self.distortion_strength = value / 1000.0
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def __init__(self):
        super().__init__()
        self.setWindowTitle("CapCut Glitch Level 2 Effect Recreation")
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        
        # Main layout
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def create_control_panel(self):
        control_widget = QWidget()
        layout = QVBoxLayout(control_widget)
        
        # File loading
        layout.addWidget(QLabel("Load Image:"))
        load_btn = QPushButton("Load Image")
        load_btn.clicked.connect(self.load_image)
        layout.addWidget(load_btn)
        
        layout.addWidget(QLabel(""))  # Spacer
        
        # Glitch Intensity
        layout.addWidget(QLabel("Glitch Intensity:"))
        self.intensity_slider = QSlider(Qt.Orientation.Horizontal)
        self.intensity_slider.setRange(0, 100)
        self.intensity_slider.setValue(80)
        self.intensity_slider.valueChanged.connect(self.gl_widget.set_glitch_intensity)
        layout.addWidget(self.intensity_slider)
        
        self.intensity_label = QLabel("80%")
        self.intensity_slider.valueChanged.connect(lambda v: self.intensity_label.setText(f"{v}%"))
        layout.addWidget(self.intensity_label)
        
        # RGB Shift
        layout.addWidget(QLabel("RGB Shift:"))
        self.rgb_slider = QSlider(Qt.Orientation.Horizontal)
        self.rgb_slider.setRange(0, 100)
        self.rgb_slider.setValue(20)
        self.rgb_slider.valueChanged.connect(self.gl_widget.set_rgb_shift)
        layout.addWidget(self.rgb_slider)
        
        self.rgb_label = QLabel("2.0%")
        self.rgb_slider.valueChanged.connect(lambda v: self.rgb_label.setText(f"{v/10.0:.1f}%"))
        layout.addWidget(self.rgb_label)
        
        # Scanlines
        layout.addWidget(QLabel("Scanlines:"))
        self.scanline_slider = QSlider(Qt.Orientation.Horizontal)
        self.scanline_slider.setRange(0, 100)
        self.scanline_slider.setValue(30)
        self.scanline_slider.valueChanged.connect(self.gl_widget.set_scanlines)
        layout.addWidget(self.scanline_slider)
        
        self.scanline_label = QLabel("30%")
        self.scanline_slider.valueChanged.connect(lambda v: self.scanline_label.setText(f"{v}%"))
        layout.addWidget(self.scanline_label)
        
        # Noise
        layout.addWidget(QLabel("Digital Noise:"))
        self.noise_slider = QSlider(Qt.Orientation.Horizontal)
        self.noise_slider.setRange(0, 100)
        self.noise_slider.setValue(10)
        self.noise_slider.valueChanged.connect(self.gl_widget.set_noise)
        layout.addWidget(self.noise_slider)
        
        self.noise_label = QLabel("10%")
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def load_image(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Image", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        # Full code available after Monday payment confirmation
        # Contact for complete implementation

