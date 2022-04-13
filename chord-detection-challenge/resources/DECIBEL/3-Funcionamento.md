#**FUNCIONAMENTO**

*   *para que possamos análisar melhor as vantagens do DECIBEL, vamos antes estudar o método tradicional de ACE, que funciona como um problema de transcrição*
    *   *De inicio são colhidos e tratados os dados de áudio.*
    *   *É feito uma segmentação do áudio em pequenos trechos,e é colhido informações delas, onde a mais comum é o chroma vector, que diz as principais notas presentes no áudio atual.*
    *   *No fim, é formado uma tabela 3D(intensidade das notas por tempo), que pode se extrair os prováveis acordes.*
  
*   *Agora será visto a forma como o DECIBEL trata o problema (de forma resumida), fazendo o modelo antigo se tornar um problema de alinhamento*
    *   *De inicio são colhidos e tratados os dados de áudio.*
    *   *Acontece então, o alinhamento do áudio com o arquivo MIDI, no meio tempo o tab file gera os acordes sem ainda a organização temporal*
    *   *Em seguida ocorre o processo mais importante, o jump align, juntando o áudio com os acordes, dando a ordem temporal*
    *   *Por fim, acontece o data fusion, juntando tudo e gerando a saída correta*