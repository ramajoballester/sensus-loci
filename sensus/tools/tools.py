class Dummy():
    """
    Dummy class for testing purposes :func:`sensus.utils.pc2pc_object`
    
    Parameters
    ----------
    a : int
        First parameter
    b : int
        Second parameter
    
    Attributes
    ----------
    a : int
        First parameter
    b : int
        Second parameter
    
    Methods
    -------
    add()
        Add two numbers together
    sub()
        Subtract two numbers
    mul()
        Multiply two numbers
    div()
        Divide two numbers

    Examples
    --------
    >>> d = Dummy(1, 2)
    >>> d.add()
    3
    """

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def add(self):
        """
        Add two numbers together

        Returns
        -------
        int
            Sum of a and b

        Raises
        ------
        ValueError
            If a or b is not int or float
        """
        if not isinstance(self.a, (int, float)) or not isinstance(self.b, (int, float)):
            raise ValueError('a and b must be int or float')
        return self.a + self.b
    
    def sub(self):
        """
        Subtract two numbers

        Returns
        -------
        int
            Difference of a and b

        Raises
        ------
        ValueError
            If a or b is not int or float
        """
        if not isinstance(self.a, (int, float)) or not isinstance(self.b, (int, float)):
            raise ValueError('a and b must be int or float')
        return self.a - self.b
    
    def mul(self):
        """
        Multiply two numbers

        Returns
        -------
        int
            Product of a and b

        Raises
        ------
        ValueError
            If a or b is not int or float
        """
        if not isinstance(self.a, (int, float)) or not isinstance(self.b, (int, float)):
            raise ValueError('a and b must be int or float')
        return self.a * self.b
    
    def div(self):
        """
        Divide two numbers

        Returns
        -------
        int
            Quotient of a and b

        Raises
        ------
        ValueError
            If a or b is not int or float
        ZeroDivisionError
            If b is zero
        """
        if not isinstance(self.a, (int, float)) or not isinstance(self.b, (int, float)):
            raise ValueError('a and b must be int or float')
        if self.b == 0:
            raise ZeroDivisionError('b must not be zero')
        return self.a / self.b