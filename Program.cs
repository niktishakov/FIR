using System;
using System.Collections.Generic;
using Accord.Statistics;

namespace FIR
{
    internal class Program
    {
        private static List<double> _input;
        private static List<double> _output;

        private static double Polyharmonic(int t)
        {
            return 30*Math.Cos(3*2*Math.PI*t) + 
                   60*Math.Cos(7*2*Math.PI*t) + 
                   90*Math.Cos(10*2*Math.PI*t) + 
                   30*Math.Cos(25*2*Math.PI*t);
        }
        private static void Run()
        {
            var fir = new Fir(128, 20, 50, 30, 30);
            _input = new List<double>();
            _output = new List<double>();

            for (var t=0; t<50; t++)
            {
                _input.Add(Polyharmonic(t));
                _output.Add(fir.Response(ref _input));
            } 
        }

        private static void PlotSolution()
        {
            var inPlot = new Accord.Statistics.Visualizations.Scatterplot("input", "X", "Y");
            inPlot.Compute(_input.ToArray());

            // var outPlot = new Accord.Statistics.Visualizations.Scatterplot("output", "X", "Y");
            // outPlot.Compute(_output.ToArray());
        }
    
        public static void Main(string[] args)
        {
            Run();
            PlotSolution();
        }
    }
}