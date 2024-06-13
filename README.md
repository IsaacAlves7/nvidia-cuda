# 👁️‍🗨️ Nvidia CUDA 

<img src="https://img.shields.io/badge/Numba-3.10.7-00A3E0?style=flat&logo=Numba&logoColor=white"> <img src="https://img.shields.io/badge/NumPy-3.10.7-00A3E0?style=flat&logo=numpy&logoColor=white"> <img src="https://img.shields.io/badge/Anaconda-3.10.7-44A833?style=flat&logo=Anaconda&logoColor=white"> <img src="https://img.shields.io/badge/CUDA-3.10.7-76B900?style=flat&logo=nvidia&logoColor=white">

<img src="https://upload.wikimedia.org/wikipedia/en/b/b9/Nvidia_CUDA_Logo.jpg" align="right" height="77">

O **Nvidia CUDA** é uma plataforma de computação paralela e um modelo de programação para ser usado com GPUs - Unidade de Processamento Gráfica. Com ele, toda a parte “pesada” do código é executada nos diversos núcleos da placa de vídeo enquanto apenas a parte sequencial do código é executada no processador, obtendo um ganho significativo de performance.

> **CUDA** anteriormente conhecida como Arquitetura de Dispositivo de Computação Unificada é uma API destinada a computação paralela.

> O uso da GPU tem custos indiretos. Se o cálculo não for pesado o suficiente, o custo (em tempo) de usar uma GPU pode ser maior do que o ganho. Por outro lado, se o cálculo for pesado, você pode ver uma grande melhoria na velocidade.

Vários termos importantes no tópico de programação CUDA estão listados aqui:

- Host: a CPU
- Device: a GPU
- Host memory: a memória principal do sistema
- Device memory: memória integrada em um cartão GPU
- Kernel: uma função GPU lançada pelo host e executada no dispositivo
- Device function: uma função GPU executada no dispositivo que só pode ser chamada a partir do dispositivo (ou seja, a partir de um kernel ou outra função do dispositivo).

<img src="https://upload.wikimedia.org/wikipedia/commons/f/fe/Numba_logo.svg" height="77" align="right">

O **Numba** oferece suporte à programação de GPU CUDA compilando diretamente um subconjunto restrito de código Python em kernels CUDA e funções de dispositivo seguindo o modelo de execução CUDA. Kernels escritos em Numba parecem ter acesso direto aos arrays **NumPy**. Matrizes NumPy são transferidas entre a CPU e a GPU automaticamente. Numba funciona permitindo que você especifique assinaturas de tipo para funções Python, o que permite a compilação em tempo de execução (isto é, “Just-in-Time” ou compilação JIT).

> **Numba** é um projeto de código aberto, licenciado por BSD, que se baseia fortemente nas capacidades do compilador LLVM.

**Exemplo**: O decorador `@vectorize`, no código a seguir, gera uma versão compilada e vetorizada da função escalar em tempo de execução para que possa ser usada para processar matrizes de dados em paralelo na GPU.

```python
import numpy as np
from numba import vectorize

@vectorize(['float32(float32, float32)'], target='cuda')
def Add(a, b):
return a + b

# Initialize arrays
N = 100000
A = np.ones(N, dtype=np.float32)
B = np.ones(A.shape, dtype=A.dtype)
C = np.empty_like(A, dtype=A.dtype)

# Add arrays on GPU
C = Add(A, B)
```

<img src="https://www.svgrepo.com/show/354127/numpy.svg" height="77" align="right">

Para compilar e executar a mesma função na CPU, simplesmente mudamos o destino para 'cpu', o que produz desempenho no nível do código C vetorizado e compilado na CPU. Essa flexibilidade ajuda a produzir código mais reutilizável e permite desenvolver em máquinas sem GPUs.

> Um dos pontos fortes da plataforma de computação paralela CUDA é a variedade de bibliotecas aceleradas por GPUs disponíveis.

Outro projeto da equipe Numba, chamado **pyculib**, fornece uma interface Python para as bibliotecas CUDA cuBLAS (álgebra linear densa), cuFFT (Fast Fourier Transform) e cuRAND (geração de número aleatório).

Muitos aplicativos serão capazes de obter uma aceleração significativa apenas usando essas bibliotecas, sem escrever nenhum código específico da GPU. Por exemplo, o código a seguir gera um milhão de números aleatórios uniformemente distribuídos na GPU usando o gerador de números pseudoaleatórios “XORWOW”.

```python
import numpy as np
from pyculib import rand as curand

prng = curand.PRNG(rndtype=curand.PRNG.XORWOW)
rand = np.empty(100000)
prng.uniform(rand)
print rand[:10]
```

<img src="https://anaconda.org/static/img/anaconda-symbol.svg" height="77" align="right">

A capacidade do Numba de compilar código dinamicamente significa que você não abre mão da flexibilidade do Python. Esse é um grande passo para fornecer a combinação ideal de programação de alta produtividade e computação de alto desempenho.

O back-end da GPU do Numba utiliza o NVIDIA Compiler SDK baseado em LLVM. Os wrappers pyculib em torno das bibliotecas CUDA também são de código aberto e licenciados por BSD.

Para começar a usar o Numba, a primeira etapa é baixar e instalar a distribuição **Anaconda Python**, uma "distribuição Python totalmente gratuita, pronta para empresas, para processamento de dados em grande escala, análise preditiva e computação científica" que inclui muitos pacotes populares (Numpy, SciPy, Matplotlib, IPython etc).

Digite o comando para baixar o Numba:

```sh
apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
```

Agora, você pode ativar a instalação, fazendo um source no arquivo `~/.bashrc: source ~/.bashrc`

Assim que tiver feito isso, você será levado ao ambiente de programação padrão de base do Anaconda, e seu prompt de comando mudará para o seguinte: `(base) summy@ubuntu:~$`

Embora o Anaconda venha com esse ambiente de programação padrão de base, você deve criar ambientes separados para seus programas e mantê-los isolados um do outro. Você pode, ainda, verificar sua instalação fazendo o uso do comando `conda`, por exemplo, com `list`:

```sh
conda list
```

Você receberá a saída de todos os pacotes disponíveis através da instalação do Anaconda.

<pre>
# packages in environment at /home/sammy/anaconda3:
# Name                    Version                  Build  Channel
_ipyw_jlab_nb_ext_conf    0.1.0                    py37_0
_libgcc_mutex             0.1                        main
alabaster                 0.7.12                   py37_0
anaconda                  2020.02                  py37_0
...
</pre>

Agora que o Anaconda está instalado, podemos seguir em frente para a configuração dos ambientes dele.

> **Atenção**: Os ambientes virtuais do Anaconda lhe permitem manter projetos organizados pelas versões do Python e pelos pacotes necessários. Para cada ambiente do Anaconda que você configurar, especifique qual versão do Python usar e mantenha todos os arquivos de programação relacionados dentro desse diretório.

Primeiro, podemos verificar quais versões do Python estão disponíveis para que possamos usar: `conda search "^python$"`

> Vamos criar um ambiente usando a versão mais recente do Python 3.

Podemos conseguir isso atribuindo a versão 3 ao argumento python. Vamos chamar o ambiente de `my_env`, mas você pode usar um nome mais descritivo para o ambiente, especialmente se estiver usando ambientes para acessar mais de uma versão do Python.

```sh
conda create --name my_env python=3
```

Você receberá uma saída com informações sobre o que está baixado e quais pacotes serão instalados e, em seguida, será solicitado a prosseguir com `y` ou `n`. Assim que concordar, digite `y`.

O utilitário `conda` agora irá obter os pacotes para o ambiente e informá-lo assim que estiver concluído. Você pode ativar seu novo ambiente digitando o seguinte:

```sh
conda activate my_env
```

Com seu ambiente ativado, seu prefixo do prompt de comando irá refletir que você não está mais no ambiente base, mas no novo ambiente que acabou de criar.

<pre>
(my_env) summy@ubuntu:~$
</pre>

Dentro do ambiente, você pode verificar se está usando a versão do Python que tinha intenção de usar: `(my_env) summy@ubuntu:~$ python –version`

Quando estiver pronto para desativar seu ambiente do Anaconda, você pode fazer isso digitando: `(my_env) summy@ubuntu:~$ conda deactivate`

Observe que pode substituir a palavra source por `.` para obter os mesmos resultados. Para focar em uma versão mais específica do Python, você pode passar uma versão específica para o argumento python, como 3.5, por exemplo:

```sh
conda create -n my_env35 python=3.5
```

Você pode inspecionar todos os ambientes que configurou com este comando:

<pre>
(base) summy@ubuntu:~$ conda info –envs


# conda environments:
#
base                  *  /home/sammy/anaconda3
my_env                   /home/sammy/anaconda3/envs/my_env
my_env35                 /home/sammy/anaconda3/envs/my_env35
</pre>

O asterisco indica o ambiente ativo atual. Cada ambiente que você criar com o `conda create` virá com vários pacotes padrão:

- `_libgcc_mutex`
- `ca-certificates`
- `certifi`
- `libedit`
- `libffi`
- `libgcc-ng`
- `libstdcxx-ng`
- `ncurses`
- `openssl`
- `pip`
- `python`
- `readline`
- `setuptools`
- `sqlite`
- `tk`
- `wheel`
- `xz`
- `zlib`

Você pode acrescentar pacotes adicionais, como o Numpy, por exemplo, com o seguinte comando:

```sh
conda install --name my_env35 numpy
```

Se você já sabe que gostaria de um ambiente Numpy após a criação, pode concentrá-lo em seu comando `conda create`:

```sh
conda create --name my_env python=3 numpy
```

Se você não estiver mais trabalhando em um projeto específico e não tiver mais necessidade do ambiente associado, pode removê-lo. Para fazer isso, digite o seguinte:

```sh
conda remove --name my_env35 --all
```

> **Atenção**: Agora, quando você digitar o comando `conda info --envs`, o ambiente que removeu não será mais listado.

Você deve garantir regularmente que o Anaconda esteja atualizado para que você esteja trabalhando com todas as versões mais recentes do pacote. Para fazer isso, deve primeiro atualizar o utilitário conda: `(base) summy@ubuntu:~$ conda update conda`

Quando solicitado a fazer isso, digite `y` para continuar com a atualização. Assim que a atualização do `conda` estiver concluída, você pode atualizar a distribuição do Anaconda:

```sh
conda update anaconda
```

> **Atenção**: Novamente, quando solicitado a fazer isso, digite `y` para continuar. Isso garantirá que você esteja usando as versões mais recentes do `conda` e do Anaconda.

Depois de instalar o Anaconda, instale os pacotes CUDA necessários digitando:

```sh
conda install numba cudatoolkit pyculib
```

> O **Anaconda** (anteriormente Continuum Analytics) reconheceu que alcançar grandes acelerações em alguns cálculos requer uma interface de programação mais expressiva com controle mais detalhado sobre o paralelismo do que as bibliotecas e a vetorização automática de `loop` podem fornecer.
>
> Portanto, o Numba possui outro conjunto importante de recursos que constitui o que é conhecido não oficialmente como “CUDA Python”.

Numba expõe o modelo de programação CUDA, assim como em CUDA C / C ++, mas usando a sintaxe Python pura, para que os programadores possam criar kernels paralelos personalizados e ajustados sem deixar o conforto e as vantagens do Python para trás. O CUDA JIT da Numba (disponível via decorador ou chamada de função) compila funções CUDA Python em tempo de execução, especializando-as para os tipos que você usa, e sua API CUDA Python fornece controle explícito sobre transferências de dados e fluxos CUDA, entre outros recursos.

O exemplo de código a seguir demonstra isso com um kernel de conjunto Mandelbrot simples. Observe que a função `mandel_kernel` usa as estruturas `cuda.threadIdx`, `cuda.blockIdx`, `cuda.blockDim` e `cuda.gridDim` fornecidas por Numba para calcular os índices globais de pixel `X` e `Y` para o segmento atual. Como em outras linguagens CUDA, lançamos o kernel inserindo uma "configuração de execução" (linguagem CUDA para o número de threads e blocos de threads a serem usados para executar o kernel) entre colchetes, entre o nome da função e a lista de argumentos: `mandel_kernel` [griddim, blockdim] (- 2.0, 1.0, -1.0, 1.0, d_image, 20). Você também pode ver o uso das funções de API `to_host` e `to_device` para copiar dados de e para a GPU.

```python
@cuda.jit
def mandel_kernel(min_x, max_x, min_y, max_y, image, iters):
	height = image.shape[0]
	width = image.shape[1]

	pixel_size_x = (max_x - min_x) / width
	pixel_size_y = (max_y - min_y) / height

	startX = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
	startY = cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y
	gridX = cuda.gridDim.x * cuda.blockDim.x;
	gridY = cuda.gridDim.y * cuda.blockDim.y;

	for x in range(startX, width, gridX):
		real = min_x + x * pixel_size_x
		for y in range(startY, height, gridY):
			imag = min_y + y * pixel_size_y
			image[y, x] = mandel(real, imag, iters)

	gimage = np.zeros((1024, 1536), dtype = np.uint8)
	blockdim = (32, 8)
	griddim = (32,16)

	start = timer()
	d_image = cuda.to_device(gimage)
	mandel_kernel[griddim, blockdim](-2.0, 1.0, -1.0, 1.0, d_image, 20)
	d_image.to_host()
	dt = timer() - start

	print "Mandelbrot created on GPU in %f s" % dt

	imshow(gimage)
```

> Em um servidor com uma GPU NVIDIA Tesla P100 e uma CPU Intel Xeon E5-2698 v3, este código CUDA Python Mandelbrot é executado quase 1700 vezes mais rápido do que a versão Python pura. 1700x pode parecer uma aceleração irreal, mas tenha em mente que estamos comparando o código Python compilado, paralelo e acelerado por GPU ao código Python interpretado de thread único na CPU.
