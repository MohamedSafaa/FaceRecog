# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models


class Result(models.Model):
	res = models.CharField(max_length=3)
	
	def __str__():
		return self.res
