# Multi-Agent-Financial-Analytics
# Agentic_AI
Agentic AI

## 🔧 How to Run the Project

### 1️⃣ **Clone the Repository**
First, download the project's source code from GitHub using the following command:

```sh
git clone https://github.com/kiencnguyen/Agentic_AI.git
```
This command will create a folder named **Agentic_AI** containing all the project's source files.

---

### 2️⃣ **Navigate to the Project Directory**
After cloning the repository, move into the project directory with the command:

```sh
cd Agentic_AI
```
This step ensures that you are in the correct folder to build and run the project.

---

### 3️⃣ **Build the Docker Image from the Dockerfile**
Next, build a Docker image from `Dockerfile.base`:

```sh
docker build -t base:latest -f Dockerfile.base .
```
- `docker build`: Command to build a new image from the Dockerfile.
- `-t base:latest`: Assigns the name (`base`) and tag (`latest`) to the image.
- `-f Dockerfile.base`: Specifies the exact Dockerfile to use for building.

Once this command runs successfully, Docker will install the necessary dependencies and create an image to run the application.

---

### 4️⃣ **Run the Project with Docker Compose**
After building the image, start the entire system using:

```sh
docker compose up --build
```
- `docker compose up`: Launches containers as defined in `docker-compose.yml`.
- `--build`: Rebuilds the image if any changes have been made in the source code.

After executing this command, the application will start and be ready for use.
