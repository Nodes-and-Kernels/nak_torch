#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# This file contains the implementation of the Himmelblau's potential
# Ayoub Belhadji
# 05/12/2025


import torch



def himmelblau(T):
    def himmelblau_aux(x, _ = None):
        x1, x2 = x[...,0], x[...,1]
        t1 = torch.square(torch.square(x1)+x2-11)
        t2 = torch.square(x1 + torch.square(x2)-7)
        Ux = t1 + t2
        return -Ux/T
    return himmelblau_aux


