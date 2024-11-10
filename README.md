# Projeto de Detecção de Medicamentos

Este projeto utiliza técnicas de visão computacional para identificar e processar imagens de medicamentos em um ambiente controlado, utilizando marcadores específicos para determinar detalhes cruciais de cada imagem capturada.

## Pré-requisitos

Antes de iniciar, certifique-se de que seu ambiente possui os seguintes softwares e bibliotecas instalados:

- Python 3.8 ou superior
- OpenCV
- NumPy
- Matplotlib
- Stag (biblioteca para detecção de marcadores)
- Rembg (para remoção de fundo das imagens)

Você também precisará de uma ferramenta para gerenciamento de pacotes como `pip`.

## Configuração do Ambiente

### Instalação das Dependências

Para instalar as dependências necessárias, você pode utilizar o `pip`. Aqui está um exemplo de como instalar as bibliotecas necessárias:

```bash
pip install opencv-python opencv-contrib-python numpy matplotlib stag-python rembg
```

### Estrutura de Diretórios

Certifique-se de que a estrutura de diretórios está configurada conforme abaixo:

```
project_directory/
│
├── frames/                # Diretório para armazenar imagens de entrada
│
└── features/              # Diretório para salvar saídas do processamento
    ├── cropped_imgs/
    ├── histogram/
    ├── medicine_png/
    ├── mask/
    └── contour/
```

## Utilização

### Carregamento e Processamento das Imagens

O script principal (`extract_features.py`) realiza as seguintes tarefas:

1. **Inicialização**: Carrega uma imagem especificada e detecta a presença de marcadores (STags).
2. **Normalização da Perspectiva**: Ajusta a perspectiva da imagem com base nos cantos identificados do marcador.
3. **Extração de Características**:
   - **Criação de Máscara**: Gera uma máscara binária do objeto de interesse.
   - **Detecção de Contornos**: Identifica e desenha os contornos baseados na máscara.
   - **Códigos de Cadeia**: Calcula e desenha os códigos de cadeia para análise de forma.
   - **Medidas de Medicamento**: Calcula dimensões específicas dos objetos detectados.
   - **Histogramas**: Gera histogramas RGB da área de interesse.

### Executando o Script

Para executar o script, use o seguinte comando no terminal:

```bash
python extract_features.py
```

Certifique-se de que a imagem de entrada esteja localizada no diretório `frames/` e que o nome do arquivo esteja corretamente especificado no script.

## Problemas Comuns

- **Escape Sequence Warning**: Se receber um aviso sobre sequências de escape no caminho do arquivo, certifique-se de usar raw strings (prefixando com `r`) ou substitua as barras invertidas (`\`) por barras normais (`/`).

- **Falha na Detecção de Marcadores**: Verifique se a imagem de entrada está clara e se os marcadores não estão obstruídos ou desfocados.

## Contribuições

Contribuições para o projeto são bem-vindas. Para contribuir, por favor, crie um fork do repositório, faça suas alterações e submeta um pull request.

## Autoria e Colaboração

**Autor:**
- **Thiago Cabral Vilar** - Desenvolvedor principal do projeto, realizado como parte da Residência em Robótica e IA.

**Auxiliar Técnico:**
- **Mateus Gonçalves Machado** - Assistência técnica no desenvolvimento e teste das funcionalidades.

**Orientador:**
- **Adrien Durand-Petiteville** - Orientação acadêmica e suporte técnico no desenvolvimento do projeto.

**Colaborador:**
- **Felipe Mendonça** - Contribuições no aperfeiçoamento das técnicas de visão computacional aplicadas.

### Contexto do Projeto

Este projeto está sendo desenvolvido como parte da monografia intitulada OP-0023('Identificação de medicamentos usando visão computacional'), conduzida sob o programa de Residência em Robótica e Inteligência Artificial. O objetivo é implementar e validar um sistema automatizado de visão computacional capaz de identificar e verificar medicamentos oncológicos durante a dispensação, utilizando algoritmos avançados em Python com bibliotecas como OpenCV, stag para detecção de marcadores, e rembg para remoção de fundo de imagens.
