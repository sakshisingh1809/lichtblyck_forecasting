# -*- coding: utf-8 -*-


class Parent:
    def __init__(self):
        # self.p5 = 'value for property p5 cannot be set'
        self.gs5 = "value for getter/setter 5 CAN be set"  # is using the property setter defined below
        self.a9 = "attribute a9 blocking method a9"

    def a1(self):
        print("parent attr a1")

    def a2(self):
        print("parent attr a2")
        # return super().__getattr__('a2') #(a) throws error (super object has no __getattr__)

    def a3(self):
        print("parent attr a3")
        # return super().__getitem__('a3') #(a) also throwns error (super object has no __getitem__)

    def a4(self):
        print("parent attr a4")
        # return super().a4()              # (b) AttributeError

    def a5(self):
        print("parent attr a5")
        # return super().a5                  # (b) AttributeError

    def a9(self):  # this function cannot be accessed
        print("parent attr a9")

    @property
    def p1(self):
        print("parent prop getter p1")

    @property
    def p2(self):
        print("parent prop getter p2")
        return super().__getattr__(
            "p4"
        )  # (a) does NOT throw error, but uses this class's __getattr__('p2') (not 'p4'!)

    @property
    def p3(self):
        print("parent prop getter p3")
        return super().__getitem__(
            "p5"
        )  # (a) does NOT throw error, but uses this class's __getATTR__('p3') (not 'p5'!)

    @property
    def p4(self):
        print("parent prop getter p4")
        return super().p4  # (!) calls this class's __getattr__

    @property
    def p5(self):
        print("parent prop getter p5")
        return super().noneexistenthere  # (!) calls this class's __getattr__

    @property
    def gs1(self):
        print("parent prop getter gs1")
        return self._gs1

    @gs1.setter
    def gs1(self, val):
        print("parent prop setter gs1")
        self._gs1 = val

    @property
    def gs2(self):
        print("parent prop getter gs2")
        return self._gs2

    @gs2.setter
    def gs2(self, val):
        print("parent prop setter gs2")
        self._gs2 = val

    @property
    def gs3(self):
        print("parent prop getter gs3")
        return self._gs3

    @gs3.setter
    def gs3(self, val):
        print("parent prop setter gs3")
        super().gs3
        self._gs3 = val

    def test_super(self):
        try:
            print("super().a1")
            super().a1
        except Exception as e:
            print(e)
        try:
            print("super().a1()")
            super().a1()
        except Exception as e:
            print(e)
        try:
            print('super()["p1"]')
            super()["p1"]
        except Exception as e:
            print(e)
        try:
            print("super().p1")
            super().p1
        except Exception as e:
            print(e)
        try:
            print("super().p1()")
            super().p1()
        except Exception as e:
            print(e)
        try:
            print('super()["p1"]')
            super()["p1"]
        except Exception as e:
            print(e)

    def __getattr__(self, name):
        if name in ["shape", "size", "__len__"]:
            return
        print("parent getATTR " + name)

    def __getitem__(self, name):
        print("parent getITEM " + name)


p = Parent()

dir(p)
p.a1()
p.a2()
p.a3()
p.a4()
p.a5()
p.p1
p.p2
p.p3
p.p4
p.p5
p.gs1
p.gs2
p.gs3
p.nonexistent
p["a1"]
p["p1"]
p["gs1"]
p["noneexistent"]
p.a1 = "new attribute a1"  # overwrites the method defined above
p.p1 = "new attribute p1"
p.gs1 = "new attribute gs1"
p.test_super()

#%%


def times2(fun):
    def wrapper():
        val = fun()
        return val * 2

    return wrapper


class Parent:
    def a(self):
        print("accessing a")
        return 4


class Test:
    def meth1(self):
        print("accessing meth1")
        return super().a()  # calling random nonexisting attribute; error (as expected)

    @times2
    def prop1(self):
        print("accessing prop1")
        return super().a()  # calling random nonexisting attribute; no error?

    def __getattr__(self, name):
        print("getattr " + name)


test = Test()
test.meth1()
test.prop1


#%% inheritance:


class Child(Parent):
    def i(self):
        print("child attr i")

    @property
    def j(self):
        print("child prop j")

    def a(self):
        print("child attr a")


c = Child()

dir(c)
c.a()
c.b
c.c()
