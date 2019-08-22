using System;
using System.Collections.Generic;

namespace FIR
{
    public class Fir
    {
        private readonly int _size; //Длина фильтра
        List<double> H;         //Импульсная характеристика фильтра
        List<double> H_d;       //Идеальная импульсная характеристика
        List<double> W;         //Весовая функция

        // omegaS - Частота дискретизации входного сигнала
        // omegaP - полоса пропускания, omegaA -  полоса заграждения
        // Aa - минимальное затухание в полосе задерживания, Ap - максимально допустимая пульсация в полосе пропускания
        // alpha - компромисс между максимальным уровнем боковых лепестков и шириной главного лепестка 
        public Fir(double omegaS, double omegaP, double omegaA, double Aa, double Ap)
        {
            
            var Fc = (omegaP + omegaA) / (2 * omegaS);
            var D = Aa <= 21 ? 0.9222 : (Aa - 7.95) / 14.36;
            var alpha = Aa <= 21 ? 
                  0.0 : Aa <= 50 ? 
                  0.5842 * Math.Pow(Aa - 21, 0.4) + 0.07886 * (Aa - 21) : 0.1102 * (Aa - 8.7);

            _size = (int)Math.Ceiling(omegaS * D / (omegaA - omegaP) * 0.5)*2 + 1;
            H = new List<double>();
            H_d = new List<double>();
            W = new List<double>();


            for (int i = -(_size-1)/2; i <= (_size-1)/2; i++)
            {
                if (Math.Abs(i) == (_size - 1) / 2) 
                    H_d.Add(0);
                    else if (i == 0) 
                        H_d.Add(2 * Math.PI * Fc);
                        else 
                            H_d.Add(Math.Sin(2 * Math.PI * Fc * i) / (Math.PI * i));
                    
                // весовая функция Кайзера
                var beta = alpha * Math.Sqrt(1 - Math.Pow(2*i/(_size-1), 2));
                W.Add(Accord.Math.Bessel.I0(beta) / Accord.Math.Bessel.I0(alpha));
                // Преобразование коэффициентов по принципу свертки 
                H.Add(H_d[H_d.Count-1] * W[W.Count-1]);
            }

            //Нормировка импульсной характеристики
            var sum = 0.0;
            for (var i = 0; i < _size; i++) sum += H[i];
            for (var i = 0; i < _size; i++) H[i] /= sum; // сумма коэффициентов равна 1
        }

        public double Response(ref List<double> input)
        {
            var output = 0.0;
            var sizeIn = input.Count;
            if (sizeIn < 1) return 0;
            
            for (var i = 0; i < _size; i++)
                if (sizeIn - 1 - i >= 0)
                {
                   output += H[i] * input[sizeIn - 1 - i];
                }

            return output;
        }
    }
}