# Snippet from: rotational_transiton.py
# Generated for client preview - Full code available after payment
# Contact for complete implementation

def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    
    # Set up OpenGL format
    format = QSurfaceFormat()
    format.setVersion(3, 3)
    format.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    format.setSamples(4)  # 4x MSAA
    QSurfaceFormat.setDefaultFormat(format)
    # Full code available after Monday payment confirmation
    # Contact for complete implementation


    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Animation properties
        self.progress = 0.0
        self.duration = 2.0  # Animation duration in seconds
        self.is_animating = False
        self.start_time = 0.0
        
        # Textures and their aspect ratios
        self.texture1: Optional[QOpenGLTexture] = None
        self.texture2: Optional[QOpenGLTexture] = None
        self.default_texture1: Optional[QOpenGLTexture] = None
        self.default_texture2: Optional[QOpenGLTexture] = None
        self.texture1_aspect = 1.0
        self.texture2_aspect = 1.0
        
        # OpenGL resources
        self.shader_program: Optional[QOpenGLShaderProgram] = None
        self.vao = None
        self.vbo_vertices = None
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def initializeGL(self):
        """Initialize OpenGL resources."""
        # Enable blending for smooth transitions
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Initialize shaders
        self.init_shaders()
        
        # Initialize geometry
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def init_shaders(self):
        """Initialize the shader program with vertex and fragment shaders."""
        self.shader_program = QOpenGLShaderProgram()
        
        # Vertex shader - simple pass-through
        vertex_shader = """
        #version 330 core
        
        layout (location = 0) in vec2 position;
        layout (location = 1) in vec2 texCoord;
        
        out vec2 fragTexCoord;
        
        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
            fragTexCoord = texCoord;
        }
        """
        
        # Fragment shader - rotational blur implementation with aspect ratio correction
        fragment_shader = """
        #version 330 core
        
        in vec2 fragTexCoord;
        out vec4 fragColor;
        
        uniform sampler2D inputTexture1;
        uniform sampler2D inputTexture2;
        uniform float progress;
        uniform vec2 resolution;
        uniform float texture1Aspect;
        uniform float texture2Aspect;
        
        #define PI 3.14159265359
        #define BLUR_SAMPLES 15
        
        // Cubic ease-in-out function
        float easeInOutCubic(float x) {
            return x < 0.5 ? 4.0 * x * x * x : 1.0 - pow(-2.0 * x + 2.0, 3.0) / 2.0;
        }
        
        // Rotate a point around a center
        vec2 rotate(vec2 point, vec2 center, float angle) {
            float cos_a = cos(angle);
            float sin_a = sin(angle);
            vec2 delta = point - center;
            return vec2(
                delta.x * cos_a - delta.y * sin_a,
                delta.x * sin_a + delta.y * cos_a
            ) + center;
        }
        
        // Apply aspect ratio correction to UV coordinates
        vec2 correctAspectRatio(vec2 uv, float textureAspect, vec2 screenResolution) {
            float screenAspect = screenResolution.x / screenResolution.y;
            
            vec2 correctedUV = uv;
            
            if (textureAspect > screenAspect) {
                // Texture is wider than screen
                float scale = screenAspect / textureAspect;
                correctedUV.y = (uv.y - 0.5) * scale + 0.5;
            } else {
                // Texture is taller than screen
                float scale = textureAspect / screenAspect;
                correctedUV.x = (uv.x - 0.5) * scale + 0.5;
            }
            
            return correctedUV;
        }
        
        // Apply rotational blur to a texture
        vec3 rotationalBlur(sampler2D tex, vec2 center, vec2 coord, float intensity, float textureAspect) {
            // Apply aspect ratio correction
            vec2 correctedCoord = correctAspectRatio(coord, textureAspect, resolution);
            vec2 correctedCenter = correctAspectRatio(center, textureAspect, resolution);
            
            vec2 delta = correctedCoord - correctedCenter;
            float radius = length(delta);
            float angle = atan(delta.y, delta.x);
            # Full code available after Monday payment confirmation
            # Contact for complete implementation


    def init_geometry(self):
        """Initialize vertex array and buffer objects."""
        # Generate VAO
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        
        # Create VBO for vertices
        self.vbo_vertices = vbo.VBO(self.vertices)
        self.vbo_vertices.bind()
        
        # Position attribute
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(0))
        
    # Full code available after Monday payment confirmation
    # Contact for complete implementation


    def create_default_textures(self):
        """Create beautiful default textures for demonstration."""
        # Create gradient textures
        size = 512
        
        # Texture 1: Radial gradient (blue to purple)
        img1 = QImage(size, size, QImage.Format.Format_RGB888)
        for y in range(size):
            for x in range(size):
                # Calculate distance from center
                dx = (x - size/2) / (size/2)
                dy = (y - size/2) / (size/2)
                dist = min(1.0, math.sqrt(dx*dx + dy*dy))
                
                # Create radial gradient
                r = int(50 + dist * 100)
                g = int(100 + dist * 50)
                b = int(200 + dist * 55)
                
                img1.setPixel(x, y, (r << 16) | (g << 8) | b)
        
        self.default_texture1 = QOpenGLTexture(img1)
        self.default_texture1.setMinificationFilter(QOpenGLTexture.Filter.Linear)
        self.default_texture1.setMagnificationFilter(QOpenGLTexture.Filter.Linear)
        self.texture1_aspect = 1.0  # Square texture
        
        # Texture 2: Spiral pattern (orange to red)
        img2 = QImage(size, size, QImage.Format.Format_RGB888)
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def paintGL(self):
        """Render the rotational blur effect."""
        glClear(GL_COLOR_BUFFER_BIT)
        
        if not self.shader_program or not self.texture1 or not self.texture2:
            return
        
        # Use shader program
        self.shader_program.bind()
        
        # Set uniforms
        self.shader_program.setUniformValue("progress", self.progress)
        self.shader_program.setUniformValue("resolution", float(self.width()), float(self.height()))
        self.shader_program.setUniformValue("texture1Aspect", self.texture1_aspect)
        self.shader_program.setUniformValue("texture2Aspect", self.texture2_aspect)
        
        # Bind textures
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def resizeGL(self, width: int, height: int):
        """Handle window resize."""
        glViewport(0, 0, width, height)
    
    # Full code available after Monday payment confirmation


    def set_progress(self, progress: float):
        """Set the transition progress (0.0 to 1.0)."""
        self.progress = max(0.0, min(1.0, progress))
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def start_animation(self):
        """Start the transition animation."""
        self.is_animating = True
        self.start_time = time.time()
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def stop_animation(self):
        """Stop the transition animation."""
        self.is_animating = False
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def update_animation(self):
        """Update animation progress."""
        if not self.is_animating:
            return
        
        elapsed = time.time() - self.start_time
        progress = elapsed / self.duration
        
    # Full code available after Monday payment confirmation
    # Contact for complete implementation


    def load_texture_from_file(self, filepath: str, texture_slot: int):
        """Load texture from image file."""
        try:
            image = QImage(filepath)
            if image.isNull():
                print(f"Failed to load image: {filepath}")
                return False
            
            # Calculate aspect ratio before conversion
            original_width = image.width()
            original_height = image.height()
            aspect_ratio = original_width / original_height
            
            # Convert to RGB format and flip vertically to correct orientation
            image = image.convertToFormat(QImage.Format.Format_RGB888)
            image = image.mirrored(False, True)  # Flip vertically
            
            # Create OpenGL texture
            texture = QOpenGLTexture(image)
            texture.setMinificationFilter(QOpenGLTexture.Filter.Linear)
            texture.setMagnificationFilter(QOpenGLTexture.Filter.Linear)
            texture.setWrapMode(QOpenGLTexture.WrapMode.ClampToEdge)
            
    # Full code available after Monday payment confirmation
    # Contact for complete implementation


    def __init__(self, parent=None):
        super().__init__(parent)
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def init_ui(self):
        """Initialize the control panel UI."""
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setMaximumHeight(150)
        
        layout = QVBoxLayout(self)
        
        # Progress control
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel("Progress:"))
        
        self.progress_slider = QSlider(Qt.Orientation.Horizontal)
        self.progress_slider.setRange(0, 1000)
        self.progress_slider.setValue(0)
        self.progress_slider.valueChanged.connect(self.on_progress_changed)
        progress_layout.addWidget(self.progress_slider)
        
        self.progress_label = QLabel("0.00")
        self.progress_label.setMinimumWidth(40)
        progress_layout.addWidget(self.progress_label)
        
        layout.addLayout(progress_layout)
        
        # Animation controls
        anim_layout = QHBoxLayout()
        
        self.play_button = QPushButton("▶ Start Animation")
        self.play_button.clicked.connect(self.on_play_clicked)
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def on_progress_changed(self, value):
        """Handle progress slider change."""
        progress = value / 1000.0
        self.progress_label.setText(f"{progress:.2f}")
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def on_play_clicked(self):
        """Handle play button click."""
        self.play_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def on_stop_clicked(self):
        """Handle stop button click."""
        self.play_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def on_reset_clicked(self):
        """Handle reset button click."""
        self.progress_slider.setValue(0)
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def on_load_texture1(self):
        """Load texture 1 from file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Texture 1", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)")
            # Full code available after Monday payment confirmation
            # Contact for complete implementation


    def on_load_texture2(self):
        """Load texture 2 from file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Texture 2", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)")
            # Full code available after Monday payment confirmation
            # Contact for complete implementation


    def set_progress_external(self, progress: float):
        """Set progress from external source (animation)."""
        self.progress_slider.blockSignals(True)
        self.progress_slider.setValue(int(progress * 1000))
        self.progress_label.setText(f"{progress:.2f}")
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def __init__(self):
        super().__init__()
        self.init_ui()
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def init_ui(self):
        """Initialize the main window UI."""
        self.setWindowTitle("Perfect Rotational Blur Transition - PyQt OpenGL")
        self.setGeometry(100, 100, 1000, 700)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        layout = QVBoxLayout(central_widget)
        
        # OpenGL widget
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def connect_signals(self):
        """Connect UI signals."""
        self.control_panel.progress_changed.connect(self.gl_widget.set_progress)
        self.control_panel.animation_started.connect(self.on_animation_started)
        self.control_panel.animation_stopped.connect(self.gl_widget.stop_animation)
        self.control_panel.load_texture1.connect(lambda f: self.gl_widget.load_texture_from_file(f, 1))
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def on_animation_started(self):
        """Handle animation start."""
        self.gl_widget.start_animation()
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def update_progress_display(self):
        """Update progress display during animation."""
        if self.gl_widget.is_animating:
            self.control_panel.set_progress_external(self.gl_widget.progress)
            if self.gl_widget.progress >= 1.0:
            # Full code available after Monday payment confirmation
            # Contact for complete implementation

