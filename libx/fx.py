import multiprocessing as mp
from typing import Iterable, Callable

def multi_proc(fun:Callable, iter:Iterable, core:int=64):
	operator = mp.Pool(core)
	res = operator.map(fun, iter)
	operator.close()
	operator.join()
	return res