using MathUtils;



double[] word_1 = [1, 0, 0];
double[] word_2 = [0, 1, 0];
double[] word_3 = [1, 1, 0];
double[] word_4 = [0, 0, 1];

double[,] W_Q = new double[,] { { 2, 0, 2 }, { 2, 0, 0 }, { 2, 1, 2 } };
double[,] W_K = new double[,] { { 2, 2, 2 }, { 0, 2, 1 }, { 0, 1, 1 } };
double[,] W_V = new double[,] { { 1, 1, 0 }, { 0, 1, 1 }, { 0, 0, 0 } };

double[] query_1 = MathX.Multiply(word_1, W_Q);
double[] key_1 = MathX.Multiply(word_1, W_K);
double[] value_1 = MathX.Multiply(word_1, W_V);

double[] query_2 = MathX.Multiply(word_2, W_Q);
double[] key_2 = MathX.Multiply(word_2, W_K);
double[] value_2 = MathX.Multiply(word_2, W_V);

double[] query_3 = MathX.Multiply(word_3, W_Q);
double[] key_3 = MathX.Multiply(word_3, W_K);
double[] value_3 = MathX.Multiply(word_3, W_V);

double[] query_4 = MathX.Multiply(word_4, W_Q);
double[] key_4 = MathX.Multiply(word_4, W_K);
double[] value_4 = MathX.Multiply(word_4, W_V);








double[] scores = [MathX.Dot(query_1, key_1), MathX.Dot(query_1, key_2), MathX.Dot(query_1, key_3), MathX.Dot(query_1, key_4)];
print(scores);

for(int i = 0; i < scores.Length; i++)
{
    scores[i] = scores[i] / Math.Sqrt(3);
}

double[] weight = MathX.Softmax(scores);
print(weight);


double[] attention = MathX.Sum(MathX.Multiply(weight[0], value_1), MathX.Multiply(weight[1], value_2));
attention = MathX.Sum(attention, MathX.Multiply(weight[2], value_3));
attention = MathX.Sum(attention, MathX.Multiply(weight[3], value_4));
print(attention);


double[] attention1 = MathX.Sum(new List<double[]>() { MathX.Multiply(weight[0], value_1), MathX.Multiply(weight[1], value_2), MathX.Multiply(weight[2], value_3), MathX.Multiply(weight[3], value_4) });




void print(double[] data)
{
    Console.Write("[");
    foreach (double x in data)
    {
        Console.Write(x + " ");
    }
    Console.Write("]");
    Console.WriteLine();
}


void printM(double[,] data)
{
    Console.Write("[");
    for (int i = 0; i < data.GetLength(0); i++)
    {
        Console.Write("[");
        for (int j = 0; j < data.GetLength(1); j++)
        {
            Console.Write(data[i,j]+" ");
        }
        Console.Write("]");
    }
    Console.Write("]");
    Console.WriteLine();
}