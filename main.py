import discord
import asyncio
import datetime
import pytz
import os
from discord.ext import tasks
from dotenv import load_dotenv
from flask import Flask


# Load environment variables safely
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
CHANNEL_ID = int(os.getenv('CHANNEL_ID'))

client = discord.Client(intents=discord.Intents.default())

app = Flask("keep_alive")  # tiny web server

@app.route("/")
def home():
    return "Bot is alive!"

# Run Flask in background thread
import threading
threading.Thread(target=lambda: app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))).start()

# Your curriculum data (same as before)
CURRICULUM = {
    1: {
        'day': 1,
        'topic': 'Vectors as Data',
        'video': '3Blue1Brown "Vectors" - https://youtu.be/fNk_zzaMoSs',
        'tasks': [
            'Install Python and NumPy: pip install numpy',
            'Create customer vectors: customer1 = np.array([25, 50000, 80])',
            'Calculate similarity: np.dot(customer1, customer2)',
            'Practice: Create 5 different customer profiles'
        ],
        'code_example': '''```python
import numpy as np
# Customer data: [age, income, spending_score]
customer1 = np.array([25, 50000, 80])
customer2 = np.array([35, 60000, 60])
similarity = np.dot(customer1, customer2)
print(f"Customer similarity: {similarity}")
```''',
        'duration': '1.5 hours',
        'resources': ['Kaggle Linear Algebra: https://kaggle.com/learn/linear-algebra']
    },
    2: {
        'day': 2,
        'topic': 'Vector Operations',
        'video': 'Vector addition/visualization - https://youtu.be/Proc6Cn1OS8',
        'tasks': [
            'Practice vector addition/subtraction',
            'Implement scalar multiplication', 
            'Build simple movie recommender system',
            'Calculate user similarities using dot products'
        ],
        'code_example': '''```python
# Vector operations practice
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
print("Addition:", v1 + v2)
print("Scalar multiplication:", 3 * v1)
```''',
        'duration': '2 hours',
        'resources': ['NumPy documentation: https://numpy.org/doc/']
    },
    3: {
        'day': 3,
        'topic': 'Matrices as Neural Network Layers',
        'video': 'Matrix multiplication - https://youtu.be/XkY2DOUCWMU',
        'tasks': [
            'Create weight matrices for neural network',
            'Implement forward pass: output = weights @ input',
            'Practice matrix-vector multiplication',
            'Build simple image transformation'
        ],
        'code_example': '''```python
# Simple neural network layer
input_data = np.array([0.5, -0.2, 0.8])
weights = np.array([[0.1, 0.4, 0.7],
                   [0.2, 0.5, 0.8],
                   [0.3, 0.6, 0.9]])
output = np.dot(weights.T, input_data)
print(f"Network output: {output}")
```''',
        'duration': '2 hours',
        'resources': ['Interactive matrix explorer: https://observablehq.com/@mbostock/matrix-explorer']
    },
    4: {
        'day': 4,
        'topic': 'Matrix Multiplication & AI',
        'video': 'Matrix transformations - https://youtu.be=XuTyb-4Ngvg',
        'tasks': [
            'Batch processing with matrix multiplication',
            'Implement mini neural network',
            'Work on Week 1 project: Movie recommender',
            'Practice with different matrix sizes'
        ],
        'code_example': '''```python
# Batch processing example
users = np.array([[1, 2, 3], [4, 5, 6]])  # 2 users
items = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])  # 3 items
predictions = users @ items
print("User-item predictions:", predictions)
```''',
        'duration': '2.5 hours',
        'resources': ['Neural Networks visualizer: https://playground.tensorflow.org/']
    },
    5: {
        'day': 5,
        'topic': 'Determinants & System Solving',
        'video': 'Determinants intuition - Use your knowledge!',
        'tasks': [
            'Check matrix invertibility with np.linalg.det()',
            'Solve simple recommendation systems',
            'Practice: When can systems be solved?',
            'Implement basic error checking'
        ],
        'code_example': '''```python
# Check if matrix is invertible
A = np.array([[1, 2], [3, 4]])
det = np.linalg.det(A)
if abs(det) > 0.0001:
    print("Matrix is invertible!")
    A_inv = np.linalg.inv(A)
else:
    print("Matrix is singular!")
```''',
        'duration': '2 hours',
        'resources': ['Linear algebra calculator: https://symbolab.com/solver/linear-algebra-calculator']
    },
    6: {
        'day': 6,
        'topic': 'Week 1 Project Day',
        'video': 'Review all Week 1 concepts',
        'tasks': [
            'Complete Movie Recommender System',
            'Add user similarity calculations',
            'Test with different datasets',
            'Prepare Week 2 learning plan'
        ],
        'code_example': '''```python
# Movie recommender system
users = np.array([[5, 3, 0, 4], [4, 0, 4, 1], [1, 1, 0, 5]])
similarity = users @ users.T
print("User similarity matrix:")
print(similarity)
```''',
        'duration': '3 hours',
        'resources': ['Project template: https://colab.research.google.com/']
    },
    7: {
        'day': 7,
        'topic': 'Review & Practice',
        'video': 'Week 1 recap session',
        'tasks': [
            'Review vectors and matrices',
            'Practice dot products and multiplications', 
            'Debug any coding issues',
            'Plan Week 2 schedule'
        ],
        'code_example': '''```python
# Practice all operations
v = np.array([1, 2, 3])
M = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Vector norm:", np.linalg.norm(v))
print("Matrix determinant:", np.linalg.det(M))
```''',
        'duration': '1.5 hours',
        'resources': ['Practice problems: https://brilliant.org/linear-algebra/']
    },
    8: {
        'day': 8,
        'topic': 'Eigenvectors & PCA',
        'video': 'Eigenvectors explained - https://youtu.be=PFDu9oVAE-g',
        'tasks': [
            'Understand "important directions" in data',
            'Implement PCA with sklearn',
            'Reduce dataset dimensions', 
            'Visualize compressed data'
        ],
        'code_example': '''```python
from sklearn.decomposition import PCA
import numpy as np

# Sample data
data = np.array([[170, 65], [180, 80], [165, 55], [175, 70]])
pca = PCA(n_components=1)
reduced = pca.fit_transform(data)
print("Reduced data shape:", reduced.shape)
```''',
        'duration': '2 hours',
        'resources': ['PCA visualizer: https://setosa.io/ev/principal-component-analysis/']
    },
    9: {
        'day': 9,
        'topic': 'Dimensionality Reduction',
        'video': 'PCA applications in AI',
        'tasks': [
            'Work with real datasets',
            'Compare original vs reduced data',
            'Implement image compression',
            'Analyze information loss'
        ],
        'code_example': '''```python
# Image compression simulation
original_data = np.random.rand(100, 100)  # Fake image
pca = PCA(n_components=50)
compressed = pca.fit_transform(original_data)
reconstructed = pca.inverse_transform(compressed)
print("Compression ratio:", compressed.size/original_data.size)
```''',
        'duration': '2.5 hours',
        'resources': ['Dataset: https://www.kaggle.com/datasets']
    },
    10: {
        'day': 10,
        'topic': 'SVD & Recommendation Systems', 
        'video': 'SVD for AI - https://youtu.be=9vdg6q9a-oI',
        'tasks': [
            'Matrix factorization with np.linalg.svd()',
            'Build advanced recommender system',
            'Implement data compression',
            'Compare with Week 1 approach'
        ],
        'code_example': '''```python
# SVD for recommendations
ratings = np.array([[5, 3, 0], [4, 0, 4], [1, 1, 5]])
U, s, Vt = np.linalg.svd(ratings)
print("Singular values:", s)
# Reconstruct with top 2 components
k = 2
approx = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
```''',
        'duration': '2.5 hours',
        'resources': ['Netflix prize explanation: https://sifter.org/simon/journal/20061211.html']
    },
    11: {
        'day': 11, 
        'topic': 'Linear Regression from Scratch',
        'video': 'Normal equation derivation',
        'tasks': [
            'Implement normal equation: (XᵀX)⁻¹Xᵀy',
            'Build house price predictor',
            'Test with different datasets',
            'Compare with sklearn implementation'
        ],
        'code_example': '''```python
# Linear regression from scratch
X = np.array([[1, 1400], [1, 1600], [1, 1700], [1, 1875]])
y = np.array([400, 430, 445, 475])
theta = np.linalg.inv(X.T @ X) @ X.T @ y
print("Model coefficients:", theta)
new_house = np.array([1, 1500])
prediction = new_house @ theta
print(f"Predicted price: ${prediction:.0f},000")
```''',
        'duration': '2 hours',
        'resources': ['Linear regression visualizer: https://www.mladdict.com/linear-regression-simulator']
    },
    12: {
        'day': 12,
        'topic': 'Gradient Descent & Optimization',
        'video': 'Gradient descent intuition', 
        'tasks': [
            'Implement basic gradient descent',
            'Compare with normal equation',
            'Practice with different learning rates',
            'Visualize convergence'
        ],
        'code_example': '''```python
# Simple gradient descent
def gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    theta = np.zeros(X.shape[1])
    for _ in range(epochs):
        gradient = X.T @ (X @ theta - y) / len(y)
        theta -= learning_rate * gradient
    return theta
```''',
        'duration': '2.5 hours',
        'resources': ['Gradient descent visualizer: https://vis.supstat.com/2013/03/gradient-descent-algorithm/']
    },
    13: {
        'day': 13,
        'topic': 'Neural Networks with Linear Algebra',
        'video': 'Neural network mathematics',
        'tasks': [
            'Build complete neural network class',
            'Implement forward/backward passes',
            'Train on simple dataset',
            'Visualize weight updates' 
        ],
        'code_example': '''```python
class SimpleNN:
    def __init__(self):
        self.W1 = np.random.randn(2, 3)
        self.W2 = np.random.randn(3, 1)
    
    def forward(self, X):
        self.z1 = X @ self.W1
        self.a1 = np.tanh(self.z1)
        self.z2 = self.a1 @ self.W2
        return self.z2
```''',
        'duration': '3 hours',
        'resources': ['Neural network visualizer: https://github.com/zjplab/nn_vis']
    },
    14: {
        'day': 14,
        'topic': 'Final Project & Review', 
        'video': 'Complete course review',
        'tasks': [
            'Complete end-to-end AI pipeline',
            'Review all concepts',
            'Plan next learning steps',
            'Celebrate completion! 🎉'
        ],
        'code_example': '''```python
# Final project: Complete pipeline
def ai_pipeline(data):
    # 1. Preprocess
    normalized = (data - data.mean()) / data.std()
    # 2. Reduce dimensions
    pca = PCA(n_components=10)
    reduced = pca.fit_transform(normalized)
    # 3. Train model
    model = SimpleNN()
    predictions = model.forward(reduced)
    return predictions
```''',
        'duration': '3 hours',
        'resources': ['Next steps: https://github.com/microsoft/ML-For-Beginners']
    }
}

class MathTutorBot(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self.current_day = 1
        self.notifications_paused = False  # ← ADD THIS
        self.pause_reason = ""  # ← ADD THIS

    async def on_ready(self):
        print(f'✅ Math Bot logged in as {self.user.name}')
        self.daily_reminder.start()
        self.reminder_9_50.start()

    # 9:50 PM Reminder
    @tasks.loop(hours=24)
    async def reminder_9_50(self):
        if self.notifications_paused:  # ← ADD THIS CHECK
            return
        await self.send_reminder(50, "🔔 **10 minutes until your math lesson!** Get ready! 🚀")

    # 10:00 PM Main Lesson
    @tasks.loop(hours=24)
    async def daily_reminder(self):
        if self.notifications_paused:  # ← ADD THIS CHECK
            return
        await self.send_reminder(0, "main")

    async def send_reminder(self, minute_offset, message_type):
        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.datetime.now(ist)
        
        if now.hour == 21 and now.minute == minute_offset:  # 9:50 PM or 10:00 PM
            channel = self.get_channel(CHANNEL_ID)
            if channel:
                if message_type == "main":
                    await self.send_daily_lesson(channel)
                else:
                    await channel.send(message_type)

    @daily_reminder.before_loop
    async def before_daily(self):
        await self.wait_until_ready()
        await self.schedule_for_ist(22, 0)  # 10:00 PM

    @reminder_9_50.before_loop
    async def before_reminder(self):
        await self.wait_until_ready()
        await self.schedule_for_ist(21, 50)  # 9:50 PM

    async def schedule_for_ist(self, target_hour, target_minute):
        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.datetime.now(ist)
        target_time = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
        
        if now >= target_time:
            target_time += datetime.timedelta(days=1)
            
        wait_seconds = (target_time - now).total_seconds()
        print(f'⏰ Next reminder in {wait_seconds/3600:.1f} hours')
        await asyncio.sleep(wait_seconds)

    async def send_daily_lesson(self, channel):
        if self.current_day <= 14:
            day_info = CURRICULUM[self.current_day]
            
            embed = discord.Embed(
                title=f"📚 Day {day_info['day']}: {day_info['topic']}",
                color=0x00ff00,
                description=f"⏰ Today's Study Plan ({day_info['duration']})"
            )
            
            embed.add_field(name="🎥 Video Lesson", value=day_info['video'], inline=False)
            
            tasks_text = "\n".join([f"• {task}" for task in day_info['tasks']])
            embed.add_field(name="💻 Coding Tasks", value=tasks_text, inline=False)
            
            embed.add_field(name="🐍 Code Example", value=day_info['code_example'], inline=False)
            
            resources_text = "\n".join(day_info['resources'])
            embed.add_field(name="📖 Resources", value=resources_text, inline=False)
            
            embed.set_footer(text=f"Progress: {self.current_day}/14 days • Keep going! 🚀")
            
            await channel.send(embed=embed)
            self.current_day = (self.current_day % 14) + 1  # Loop after day 14

    async def on_message(self, message):
        if message.author == self.user:
            return

        content = message.content.lower()
        
        # ADD PAUSE/RESUME/STATUS COMMANDS HERE
        if content == 'pause':
            self.notifications_paused = True
            self.pause_reason = "Manually paused by user"
            await message.channel.send("⏸️ **Notifications PAUSED**\nUse `resume` to restart daily lessons.")
            return

        elif content == 'resume':
            self.notifications_paused = False
            self.pause_reason = ""
            await message.channel.send("▶️ **Notifications RESUMED**\nDaily lessons will continue from Day " + str(self.current_day))
            return

        elif content == 'status':
            if self.notifications_paused:
                status_msg = f"⏸️ **PAUSED** - {self.pause_reason}\nCurrent Day: {self.current_day}"
            else:
                status_msg = f"🔔 **ACTIVE** - Next lesson: Day {self.current_day}\nNext reminder at 10:00 PM IST"
            await message.channel.send(f"🤖 **Bot Status:** {status_msg}")
            return

        # YOUR EXISTING COMMANDS
        elif content.startswith('day '):
            try:
                day_num = int(content.split()[1])
                if 1 <= day_num <= 14:
                    self.current_day = day_num
                    channel = self.get_channel(CHANNEL_ID)
                    if channel:
                        await self.send_daily_lesson(channel)
            except (IndexError, ValueError):
                await message.channel.send("Usage: `day 1` to `day 14`")

        elif content == 'progress':
            await message.channel.send(f"📊 **Current Progress:** Day {self.current_day-1}/14")

        elif content == 'help':
            help_text = """
**Commands:**
`day X` - Get specific day's lesson (1-14)
`progress` - Check your progress
`pause` - Stop automatic notifications
`resume` - Restart notifications  
`status` - Check bot status
`help` - This message commands

**Auto-schedule:**
9:50 PM - Preparation reminder
10:00 PM - Full lesson delivery
"""
            await message.channel.send(help_text)

# Run the bot
client = MathTutorBot()
client.run(TOKEN)



